/*
 * Copyright 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS

#include "BlurFilter.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <ui/GraphicTypes.h>
#include <cstdint>

#include <utils/Trace.h>

namespace android {
namespace renderengine {
namespace gl {

BlurFilter::BlurFilter(GLESRenderEngine& engine)
      : mEngine(engine),
        mCompositionFbo(engine),
        mPingFbo(engine),
        mPongFbo(engine),
        mMixProgram(engine),
        mDownsampleProgram(engine),
        mUpsampleProgram(engine) {
    mMixProgram.compile(getVertexShader(), getMixFragShader());
    mMPosLoc = mMixProgram.getAttributeLocation("aPosition");
    mMUvLoc = mMixProgram.getAttributeLocation("aUV");
    mMTextureLoc = mMixProgram.getUniformLocation("uTexture");
    mMCompositionTextureLoc = mMixProgram.getUniformLocation("uCompositionTexture");
    mMMixLoc = mMixProgram.getUniformLocation("uMix");
    mMScaleLoc = mMixProgram.getUniformLocation("uScale");

    mDownsampleProgram.compile(getVertexShader(), getDownsampleFragShader());
    mDPosLoc = mDownsampleProgram.getAttributeLocation("aPosition");
    mDUvLoc = mDownsampleProgram.getAttributeLocation("aUV");
    mDTextureLoc = mDownsampleProgram.getUniformLocation("uTexture");
    mDOffsetLoc = mDownsampleProgram.getUniformLocation("uOffset");
    mDHalfPixelLoc = mDownsampleProgram.getUniformLocation("uHalfPixel");
    mDScaleLoc = mDownsampleProgram.getUniformLocation("uScale");

    mUpsampleProgram.compile(getVertexShader(), getUpsampleFragShader());
    mUPosLoc = mUpsampleProgram.getAttributeLocation("aPosition");
    mUUvLoc = mUpsampleProgram.getAttributeLocation("aUV");
    mUTextureLoc = mUpsampleProgram.getUniformLocation("uTexture");
    mUOffsetLoc = mUpsampleProgram.getUniformLocation("uOffset");
    mUHalfPixelLoc = mUpsampleProgram.getUniformLocation("uHalfPixel");
    mUScaleLoc = mUpsampleProgram.getUniformLocation("uScale");

    static constexpr auto size = 2.0f;
    static constexpr auto translation = 1.0f;
    const GLfloat vboData[] = {
        // Vertex data
        translation - size, -translation - size,
        translation - size, -translation + size,
        translation + size, -translation + size,
        // UV data
        0.0f, 0.0f - translation,
        0.0f, size - translation,
        size, size - translation
    };
    mMeshBuffer.allocateBuffers(vboData, 12 /* size */);
}

status_t BlurFilter::setAsDrawTarget(const DisplaySettings& display, uint32_t radius) {
    ATRACE_NAME("BlurFilter::setAsDrawTarget");
    mRadius = radius;
    mDisplayX = display.physicalDisplay.left;
    mDisplayY = display.physicalDisplay.top;

    if (mDisplayWidth < display.physicalDisplay.width() ||
        mDisplayHeight < display.physicalDisplay.height()) {
        ATRACE_NAME("BlurFilter::allocatingTextures");

        mDisplayWidth = display.physicalDisplay.width();
        mDisplayHeight = display.physicalDisplay.height();
        mCompositionFbo.allocateBuffers(mDisplayWidth, mDisplayHeight);

        const uint32_t fboWidth = floorf(mDisplayWidth * kFboScale);
        const uint32_t fboHeight = floorf(mDisplayHeight * kFboScale);
        mPingFbo.allocateBuffers(fboWidth, fboHeight);
        mPongFbo.allocateBuffers(fboWidth, fboHeight);

        if (mPingFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
            ALOGE("Invalid ping buffer");
            return mPingFbo.getStatus();
        }
        if (mPongFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
            ALOGE("Invalid pong buffer");
            return mPongFbo.getStatus();
        }
        if (mCompositionFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
            ALOGE("Invalid composition buffer");
            return mCompositionFbo.getStatus();
        }
        if (!mDownsampleProgram.isValid()) {
            ALOGE("Invalid downsample shader");
            return GL_INVALID_OPERATION;
        }
        if (!mUpsampleProgram.isValid()) {
            ALOGE("Invalid upsample shader");
            return GL_INVALID_OPERATION;
        }
    }

    mCompositionFbo.bind();
    glViewport(0, 0, mCompositionFbo.getBufferWidth(), mCompositionFbo.getBufferHeight());
    return NO_ERROR;
}

void BlurFilter::drawMesh(GLuint uv, GLuint position) {
    glEnableVertexAttribArray(uv);
    glEnableVertexAttribArray(position);
    // TODO: bind around drawMesh regions
    mMeshBuffer.bind();
    glVertexAttribPointer(position, 2 /* size */, GL_FLOAT, GL_FALSE,
                          2 * sizeof(GLfloat) /* stride */, 0 /* offset */);
    glVertexAttribPointer(uv, 2 /* size */, GL_FLOAT, GL_FALSE, 0 /* stride */,
                          (GLvoid*)(6 * sizeof(GLfloat)) /* offset */);
    mMeshBuffer.unbind();

    // draw mesh
    glDrawArrays(GL_TRIANGLES, 0 /* first */, 3 /* count */);
}

status_t BlurFilter::prepare() {
    ATRACE_NAME("BlurFilter::prepare");

    // Kawase is an approximation of Gaussian, but it behaves differently from it.
    // A radius transformation is required for approximating them, and also to introduce
    // non-integer steps, necessary to smoothly interpolate large radii.
    //const auto radius = mRadius / 6.0f;

    // Calculate how many passes we'll do, based on the radius.
    const auto passes = 5.0f;

    //const float radiusByPasses = radius / (float)passes;
    const float stepX = 7.25f;
    const float stepY = stepX;

    auto sourceWidth = mPingFbo.getBufferWidth();
    auto sourceHeight = mPingFbo.getBufferHeight();
    auto targetWidth = sourceWidth;
    auto targetHeight = sourceHeight;

    auto scale = 1.0f;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());

    GLFramebuffer* read = &mPingFbo;
    GLFramebuffer* draw = &mPongFbo;

    ALOGI("BLUR: prepare - initial dims %dx%d", targetWidth, targetHeight);

    // Set up downsampling shader
    mDownsampleProgram.useProgram();
    glUniform1i(mDTextureLoc, 0);

    // Downsample
    for (auto i = 0; i < passes; i++) {
        // Reduce resolution every pass
        targetWidth /= 2;
        targetHeight /= 2;
        scale /= 2;

        ATRACE_NAME("BlurFilter::renderDownsamplePass");
        ALOGI("BLUR: downsample to %dx%d, scale %f", targetWidth, targetHeight, scale);
        // Viewport = dest
        glViewport(0, 0, targetWidth, targetHeight);
        // Vertex shader scale = src
        glUniform1f(mDScaleLoc, scale);
        draw->bind();

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        glUniform2f(mDOffsetLoc, stepX, stepY);
        glUniform2f(mDHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mDUvLoc, mDPosLoc);

        // Swap buffers for next iteration
        std::swap(draw, read);
    }

    // Set up upsampling shader
    mUpsampleProgram.useProgram();
    glUniform1i(mUTextureLoc, 0);

    // Upsample
    for (auto i = 0; i < passes; i++) {
        // Increase resolution
        targetWidth *= 2;
        targetHeight *= 2;
        scale *= 2;

        ATRACE_NAME("BlurFilter::renderUpsamplePass");
        ALOGI("BLUR: upsample to %dx%d, scale %f", targetWidth, targetHeight, scale);
        // Viewport = dest
        glViewport(0, 0, targetWidth, targetHeight);
        // Vertex shader scale = src
        glUniform1f(mUScaleLoc, scale);
        draw->bind();

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        glUniform2f(mUOffsetLoc, stepX, stepY);
        glUniform2f(mUHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mUUvLoc, mUPosLoc);

        // Swap buffers for next iteration
        std::swap(draw, read);
    }

    ALOGI("BLUR: final target dims %dx%d, scale %f", targetWidth, targetHeight, scale);

    mLastDrawTarget = read;
    return NO_ERROR;
}

status_t BlurFilter::render(bool multiPass) {
    ATRACE_NAME("BlurFilter::render");

    // Now let's scale our blur up. It will be interpolated with the larger composited
    // texture for the first frames, to hide downscaling artifacts.
    GLfloat mix = fmin(1.0, mRadius / kMaxCrossFadeRadius);

    // When doing multiple passes, we cannot try to read mCompositionFbo, given that we'll
    // be writing onto it. Let's disable the crossfade, otherwise we'd need 1 extra frame buffer,
    // as large as the screen size.
    if (mix >= 1 || multiPass) {
        mLastDrawTarget->bindAsReadBuffer();
        glBlitFramebuffer(0, 0, mLastDrawTarget->getBufferWidth(),
                          mLastDrawTarget->getBufferHeight(), mDisplayX, mDisplayY, mDisplayWidth,
                          mDisplayHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        return NO_ERROR;
    }

    // Crossfade
    mMixProgram.useProgram();
    glUniform1f(mMMixLoc, mix);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mLastDrawTarget->getTextureName());
    glUniform1i(mMTextureLoc, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());
    glUniform1i(mMCompositionTextureLoc, 1);

    drawMesh(mMUvLoc, mMPosLoc);

    // Clean up
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
    mEngine.checkErrors("Drawing blur mesh");
    return NO_ERROR;
}

string BlurFilter::getVertexShader() const {
    return R"SHADER(#version 310 es
        precision mediump float;

        in highp float uScale;
        in vec2 aPosition;
        in highp vec2 aUV;
        out highp vec2 vUV;

        void main() {
            vUV = aUV;
            gl_Position = vec4(aPosition * uScale, 0.0, 1.0);
        }
    )SHADER";
}

string BlurFilter::getDownsampleFragShader() const {
    return R"SHADER(#version 310 es
        precision mediump float;

        uniform sampler2D uTexture;
        uniform vec2 uOffset;
        uniform vec2 uHalfPixel;

        in highp vec2 vUV;
        out vec4 fragColor;

        void main() {
            vec4 sum = texture(uTexture, vUV) * 4.0;
            sum += texture(uTexture, vUV - uHalfPixel.xy * uOffset);
            sum += texture(uTexture, vUV + uHalfPixel.xy * uOffset);
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x, -uHalfPixel.y) * uOffset);
            sum += texture(uTexture, vUV - vec2(uHalfPixel.x, -uHalfPixel.y) * uOffset);
            fragColor = sum / 8.0;
        }
    )SHADER";
}

string BlurFilter::getUpsampleFragShader() const {
    return R"SHADER(#version 310 es
        precision mediump float;

        uniform sampler2D uTexture;
        uniform vec2 uOffset;
        uniform vec2 uHalfPixel;

        in highp vec2 vUV;
        out vec4 fragColor;

        void main() {
            vec4 sum = texture(uTexture, vUV + vec2(-uHalfPixel.x * 2.0, 0.0));
            sum += texture(uTexture, vUV + vec2(-uHalfPixel.x, uHalfPixel.y)) * 2.0;
            sum += texture(uTexture, vUV + vec2(0.0, uHalfPixel.y * 2.0));
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x, uHalfPixel.y)) * 2.0;
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x * 2.0, 0.0));
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x, -uHalfPixel.y)) * 2.0;
            sum += texture(uTexture, vUV + vec2(0.0, -uHalfPixel.y * 2.0));
            sum += texture(uTexture, vUV + vec2(-uHalfPixel.x, -uHalfPixel.y)) * 2.0;
            fragColor = sum / 12.0;
        }
    )SHADER";
}

string BlurFilter::getMixFragShader() const {
    string shader = R"SHADER(#version 310 es
        precision mediump float;

        in highp vec2 vUV;
        out vec4 fragColor;

        uniform sampler2D uCompositionTexture;
        uniform sampler2D uTexture;
        uniform float uMix;

        void main() {
            vec4 blurred = texture(uTexture, vUV);
            vec4 composition = texture(uCompositionTexture, vUV);
            fragColor = mix(composition, blurred, uMix);
        }
    )SHADER";
    return shader;
}

} // namespace gl
} // namespace renderengine
} // namespace android

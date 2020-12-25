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
        mPassFbos(),
        mMixProgram(engine),
        mDownsampleProgram(engine),
        mUpsampleProgram(engine) {
    ALOGI("SARU: ----------------------------------------------------- CREATE FILTER ----------------------------------------");
    mMixProgram.compile(getVertexShader(), getMixFragShader());
    mMPosLoc = mMixProgram.getAttributeLocation("aPosition");
    mMUvLoc = mMixProgram.getAttributeLocation("aUV");
    mMTextureLoc = mMixProgram.getUniformLocation("uTexture");
    mMCompositionTextureLoc = mMixProgram.getUniformLocation("uCompositionTexture");
    mMMixLoc = mMixProgram.getUniformLocation("uMix");

    mDownsampleProgram.compile(getVertexShader(), getDownsampleFragShader());
    mDPosLoc = mDownsampleProgram.getAttributeLocation("aPosition");
    mDUvLoc = mDownsampleProgram.getAttributeLocation("aUV");
    mDTextureLoc = mDownsampleProgram.getUniformLocation("uTexture");
    mDOffsetLoc = mDownsampleProgram.getUniformLocation("uOffset");
    mDHalfPixelLoc = mDownsampleProgram.getUniformLocation("uHalfPixel");

    mUpsampleProgram.compile(getVertexShader(), getUpsampleFragShader());
    mUPosLoc = mUpsampleProgram.getAttributeLocation("aPosition");
    mUUvLoc = mUpsampleProgram.getAttributeLocation("aUV");
    mUTextureLoc = mUpsampleProgram.getUniformLocation("uTexture");
    mUOffsetLoc = mUpsampleProgram.getUniformLocation("uOffset");
    mUHalfPixelLoc = mUpsampleProgram.getUniformLocation("uHalfPixel");

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

    // Calculate passes and offsets here instead of propagating radius
    mRadius = radius;
    mPasses = 3;
    mOffset = 6.f;

    mDisplayX = display.physicalDisplay.left;
    mDisplayY = display.physicalDisplay.top;

    if (mDisplayWidth < display.physicalDisplay.width() ||
        mDisplayHeight < display.physicalDisplay.height()) {
        ATRACE_NAME("BlurFilter::allocatingTextures");
        ALOGI("SARU: ----------------------------------------------------- NEW FILTER TARGET, alloc textures");

        mDisplayWidth = display.physicalDisplay.width();
        mDisplayHeight = display.physicalDisplay.height();
        mCompositionFbo.allocateBuffers(mDisplayWidth, mDisplayHeight);

        if (mPassFbos.size() > 0) {
            for (auto fbo : mPassFbos) {
                delete fbo;
            }
        }

        const uint32_t sourceFboWidth = floorf(mDisplayWidth * kFboScale);
        const uint32_t sourceFboHeight = floorf(mDisplayHeight * kFboScale);
        for (int i = 0; i < mPasses + 1; i++) {
            // FIXME: memory leak on filter destroy
            GLFramebuffer* fbo = new GLFramebuffer(mEngine);
            ALOGI("SARU: alloc texture %dx%d", sourceFboWidth >> i, sourceFboHeight >> i);
            fbo->allocateBuffers(sourceFboWidth >> i, sourceFboHeight >> i);
            if (fbo->getStatus() != GL_FRAMEBUFFER_COMPLETE) {
                ALOGE("Invalid pass buffer");
                return fbo->getStatus();
            }

            mPassFbos.push_back(fbo);
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

    auto sourceWidth = mPassFbos[0]->getBufferWidth();
    auto sourceHeight = mPassFbos[0]->getBufferHeight();
    auto targetWidth = sourceWidth;
    auto targetHeight = sourceHeight;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());

    ALOGI("SARU: prepare - initial dims %dx%d", targetWidth, targetHeight);

    // Set up downsampling shader
    mDownsampleProgram.useProgram();
    glUniform1i(mDTextureLoc, 0);

    GLFramebuffer* read;
    GLFramebuffer* draw;

    // Downsample
    // 0 -> 1 -> ...
    for (auto i = 0; i < mPasses; i++) {
        // Reduce resolution every pass
        targetWidth /= 2;
        targetHeight /= 2;

        ATRACE_NAME("BlurFilter::renderDownsamplePass");
        ALOGI("SARU: downsample to %dx%d", targetWidth, targetHeight);
        // Viewport = dest
        glViewport(0, 0, targetWidth, targetHeight);

        // In downsampling, we never sample from fbo 0 to avoid an unnecessary blit
        draw = mPassFbos[i + 1];
        if (i == 0) {
            read = &mCompositionFbo;
        } else {
            read = mPassFbos[i];
        }

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        draw->bind();

        glUniform2f(mDOffsetLoc, mOffset, mOffset);
        glUniform2f(mDHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mDUvLoc, mDPosLoc);
    }

    // Set up upsampling shader
    mUpsampleProgram.useProgram();
    glUniform1i(mUTextureLoc, 0);

    // Upsample
    for (auto i = 0; i < mPasses; i++) {
        // Increase resolution
        targetWidth *= 2;
        targetHeight *= 2;

        ATRACE_NAME("BlurFilter::renderUpsamplePass");
        ALOGI("SARU: upsample to %dx%d", targetWidth, targetHeight);
        // Viewport = dest
        glViewport(0, 0, targetWidth, targetHeight);

        read = mPassFbos[mPasses - i];
        draw = mPassFbos[mPasses - i - 1];

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        draw->bind();

        glUniform2f(mUOffsetLoc, mOffset, mOffset);
        glUniform2f(mUHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mUUvLoc, mUPosLoc);
    }

    ALOGI("SARU: final target dims %dx%d", targetWidth, targetHeight);

    // FIXME: early termination
    /*
    mLastDrawTarget = read;
    if (scale == 1.0f) {
    return NO_ERROR;
    }
    */
    // We want the last draw buffer, but it was just swapped at the end of the last iteration, so use read
    mLastDrawTarget = draw;
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

        in vec2 aPosition;
        in highp vec2 aUV;
        out highp vec2 vUV;

        void main() {
            vUV = aUV;
            gl_Position = vec4(aPosition, 0.0, 1.0);
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
            fragColor = vec4(sum.rgb / 8.0, 1.0);
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
            vec4 sum = texture(uTexture, vUV + vec2(-uHalfPixel.x * 2.0, 0.0) * uOffset);
            sum += texture(uTexture, vUV + vec2(-uHalfPixel.x, uHalfPixel.y) * uOffset) * 2.0;
            sum += texture(uTexture, vUV + vec2(0.0, uHalfPixel.y * 2.0) * uOffset);
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x, uHalfPixel.y) * uOffset) * 2.0;
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x * 2.0, 0.0) * uOffset);
            sum += texture(uTexture, vUV + vec2(uHalfPixel.x, -uHalfPixel.y) * uOffset) * 2.0;
            sum += texture(uTexture, vUV + vec2(0.0, -uHalfPixel.y * 2.0) * uOffset);
            sum += texture(uTexture, vUV + vec2(-uHalfPixel.x, -uHalfPixel.y) * uOffset) * 2.0;
            fragColor = vec4(sum.rgb / 12.0, 1.0);
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

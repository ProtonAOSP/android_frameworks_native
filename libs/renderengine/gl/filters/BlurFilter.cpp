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
        mDitherFbo(engine),
        mMixProgram(engine),
        mDownsampleProgram(engine),
        mUpsampleProgram(engine) {
    // Create VBO first for usage in shader VAOs
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

    mMixProgram.compile(getVertexShader(), getMixFragShader());
    mMPosLoc = mMixProgram.getAttributeLocation("aPosition");
    mMUvLoc = mMixProgram.getAttributeLocation("aUV");
    mMCompositionTextureLoc = mMixProgram.getUniformLocation("uCompositionTexture");
    mMBlurredTextureLoc = mMixProgram.getUniformLocation("uBlurredTexture");
    mMDitherTextureLoc = mMixProgram.getUniformLocation("uDitherTexture");
    mMBlurOpacityLoc = mMixProgram.getUniformLocation("uBlurOpacity");
    createVertexArray(&mMVertexArray, mMPosLoc, mMUvLoc);

    mDownsampleProgram.compile(getVertexShader(), getDownsampleFragShader());
    mDPosLoc = mDownsampleProgram.getAttributeLocation("aPosition");
    mDUvLoc = mDownsampleProgram.getAttributeLocation("aUV");
    mDTextureLoc = mDownsampleProgram.getUniformLocation("uTexture");
    mDOffsetLoc = mDownsampleProgram.getUniformLocation("uOffset");
    mDHalfPixelLoc = mDownsampleProgram.getUniformLocation("uHalfPixel");
    createVertexArray(&mDVertexArray, mDPosLoc, mDUvLoc);

    mUpsampleProgram.compile(getVertexShader(), getUpsampleFragShader());
    mUPosLoc = mUpsampleProgram.getAttributeLocation("aPosition");
    mUUvLoc = mUpsampleProgram.getAttributeLocation("aUV");
    mUTextureLoc = mUpsampleProgram.getUniformLocation("uTexture");
    mUOffsetLoc = mUpsampleProgram.getUniformLocation("uOffset");
    mUHalfPixelLoc = mUpsampleProgram.getUniformLocation("uHalfPixel");
    createVertexArray(&mUVertexArray, mUPosLoc, mUUvLoc);

    const GLubyte bayerPattern[] = {
          0, 128,  32, 160,   8, 136,  40, 168,
        192,  64, 224,  96, 200,  72, 232, 104,
         48, 176,  16, 144,  56, 184,  24, 152,
        240, 112, 208,  80, 248, 120, 216,  88,
         12, 140,  44, 172,   4, 132,  36, 164,
        204,  76, 236, 108, 196,  68, 228, 100,
         60, 188,  28, 156,  52, 180,  20, 148,
        252, 124, 220,  92, 244, 116, 212,  84,
    };
    mDitherFbo.allocateBuffers(8, 8, (void *) bayerPattern,
                               GL_NEAREST, GL_REPEAT,
                               GL_R8, GL_RED);
}

status_t BlurFilter::setAsDrawTarget(const DisplaySettings& display, uint32_t radius) {
    ATRACE_NAME("BlurFilter::setAsDrawTarget");

    mDisplayX = display.physicalDisplay.left;
    mDisplayY = display.physicalDisplay.top;

    if (mDisplayWidth < display.physicalDisplay.width() ||
        mDisplayHeight < display.physicalDisplay.height()) {
        ATRACE_NAME("BlurFilter::allocatingTextures");
        ALOGI("SARU: ----------------------------------------------------- NEW FILTER TARGET, alloc textures");

        mDisplayWidth = display.physicalDisplay.width();
        mDisplayHeight = display.physicalDisplay.height();
        mCompositionFbo.allocateBuffers(mDisplayWidth, mDisplayHeight);
        if (mCompositionFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
            ALOGE("Invalid composition buffer");
            return mCompositionFbo.getStatus();
        }

        if (mPassFbos.size() > 0) {
            for (auto fbo : mPassFbos) {
                // FIXME: delete texture
                delete fbo;
            }
        }

        const uint32_t sourceFboWidth = floorf(mDisplayWidth * kFboScale);
        const uint32_t sourceFboHeight = floorf(mDisplayHeight * kFboScale);
        uint32_t allocPasses = mPasses;
        // FIXME
        // TODO: max passes for resolution
        allocPasses = 6;
        for (auto i = 0; i < allocPasses + 1; i++) {
            // FIXME: memory leak on filter destroy
            GLFramebuffer* fbo = new GLFramebuffer(mEngine);

            ALOGI("SARU: alloc texture %dx%d", sourceFboWidth >> i, sourceFboHeight >> i);
            fbo->allocateBuffers(sourceFboWidth >> i, sourceFboHeight >> i, nullptr,
                                 GL_LINEAR, GL_MIRRORED_REPEAT,
                                 GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV);

            if (fbo->getStatus() != GL_FRAMEBUFFER_COMPLETE) {
                ALOGE("Invalid pass buffer");
                return fbo->getStatus();
            }

            mPassFbos.push_back(fbo);
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

    // Approximate Gaussian blur radius
    mRadius = radius;
    // FIXME
    const uint32_t fboWidth = floorf(mDisplayWidth * kFboScale);
    mPasses = (uint32_t) ceil(std::sqrt(radius) / 5.0f);
    mOffset = ((float) fboWidth) / ((float) mPasses * (float) mPasses) / 7.5f;
    ALOGI("SARU: ---------------------------------- new radius: %d  : passes=%d offset=%f", radius, mPasses, mOffset);

    mCompositionFbo.bind();
    glViewport(0, 0, mCompositionFbo.getBufferWidth(), mCompositionFbo.getBufferHeight());
    return NO_ERROR;
}

void BlurFilter::createVertexArray(GLuint* vertexArray, GLuint position, GLuint uv) {
    glGenVertexArrays(1, vertexArray);
    glBindVertexArray(*vertexArray);
    mMeshBuffer.bind();

    glEnableVertexAttribArray(position);
    glVertexAttribPointer(position, 2 /* size */, GL_FLOAT, GL_FALSE,
                          2 * sizeof(GLfloat) /* stride */, 0 /* offset */);

    glEnableVertexAttribArray(uv);
    glVertexAttribPointer(uv, 2 /* size */, GL_FLOAT, GL_FALSE, 0 /* stride */,
                          (GLvoid*)(6 * sizeof(GLfloat)) /* offset */);

    mMeshBuffer.unbind();
    glBindVertexArray(0);
}

void BlurFilter::drawMesh(GLuint vertexArray) {
    glBindVertexArray(vertexArray);
    glDrawArrays(GL_TRIANGLES, 0 /* first */, 3 /* vertices */);
    glBindVertexArray(0);
}

status_t BlurFilter::prepare() {
    ATRACE_NAME("BlurFilter::prepare");

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());

    ALOGI("SARU: prepare - initial dims %dx%d", mPassFbos[0]->getBufferWidth(), mPassFbos[0]->getBufferHeight());

    // Set up downsampling shader
    mDownsampleProgram.useProgram();
    glUniform1i(mDTextureLoc, 0);

    GLFramebuffer* read;
    GLFramebuffer* draw;

    // Downsample
    for (auto i = 0; i < mPasses; i++) {
        ATRACE_NAME("BlurFilter::renderDownsamplePass");

        // Skip FBO 0 to avoid unnecessary blit
        draw = mPassFbos[i + 1];
        if (i == 0) {
            read = &mCompositionFbo;
        } else {
            read = mPassFbos[i];
        }

        auto targetWidth = draw->getBufferWidth();
        auto targetHeight = draw->getBufferHeight();
        glViewport(0, 0, targetWidth, targetHeight);

        ALOGI("SARU: downsample to %dx%d", targetWidth, targetHeight);

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        draw->bind();

        glUniform1f(mDOffsetLoc, mOffset);
        glUniform2f(mDHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mDVertexArray);
    }

    // Set up upsampling shader
    mUpsampleProgram.useProgram();
    glUniform1i(mUTextureLoc, 0);

    // Upsample
    for (auto i = 0; i < mPasses; i++) {
        ATRACE_NAME("BlurFilter::renderUpsamplePass");

        // Upsampling goes in the reverse direction
        read = mPassFbos[mPasses - i];
        draw = mPassFbos[mPasses - i - 1];

        auto targetWidth = draw->getBufferWidth();
        auto targetHeight = draw->getBufferHeight();
        glViewport(0, 0, targetWidth, targetHeight);

        ALOGI("SARU: upsample to %dx%d", targetWidth, targetHeight);

        glBindTexture(GL_TEXTURE_2D, read->getTextureName());
        draw->bind();

        glUniform1f(mUOffsetLoc, mOffset);
        // 1/2 pixel size in NDC
        glUniform2f(mUHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mUVertexArray);
    }

    mLastDrawTarget = draw;
    return NO_ERROR;
}

status_t BlurFilter::render(bool /*multiPass*/) {
    ATRACE_NAME("BlurFilter::render");

    // Now let's scale our blur up. It will be interpolated with the larger composited
    // texture for the first frames, to hide downscaling artifacts.
    GLfloat opacity = fmin(1.0, mRadius / kMaxCrossFadeRadius);

    // When doing multiple passes, we cannot try to read mCompositionFbo, given that we'll
    // be writing onto it. Let's disable the crossfade, otherwise we'd need 1 extra frame buffer,
    // as large as the screen size.
    //if (opacity >= 1 || multiPass) {
    //    mLastDrawTarget->bindAsReadBuffer();
    //    glBlitFramebuffer(0, 0, mLastDrawTarget->getBufferWidth(),
    //                      mLastDrawTarget->getBufferHeight(), mDisplayX, mDisplayY, mDisplayWidth,
    //                      mDisplayHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    //    return NO_ERROR;
    //}

    // Crossfade using mix shader
    mMixProgram.useProgram();
    glUniform1f(mMBlurOpacityLoc, opacity);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());
    glUniform1i(mMCompositionTextureLoc, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mLastDrawTarget->getTextureName());
    glUniform1i(mMBlurredTextureLoc, 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mDitherFbo.getTextureName());
    glUniform1i(mMDitherTextureLoc, 2);

    drawMesh(mMVertexArray);

    // Clean up to avoid breaking further composition
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
    mEngine.checkErrors("Mixing blur");
    return NO_ERROR;
}

string BlurFilter::getVertexShader() const {
    return R"SHADER(
        #version 310 es
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
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uTexture;
        uniform float uOffset;
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
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uTexture;
        uniform float uOffset;
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
            fragColor = sum / 12.0;
        }
    )SHADER";
}

string BlurFilter::getMixFragShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uCompositionTexture;
        uniform sampler2D uBlurredTexture;
        uniform sampler2D uDitherTexture;
        uniform float uBlurOpacity;

        in highp vec2 vUV;
        out vec4 fragColor;

        void main() {
            vec4 blurred = texture(uBlurredTexture, vUV);
            vec4 composition = texture(uCompositionTexture, vUV);

            float ditherLum = texture(uDitherTexture, gl_FragCoord.xy / 8.0).r;
            float dither = ditherLum / 32.0 - (1.0 / 128.0);
            blurred += dither;

            fragColor = mix(composition, blurred, uBlurOpacity);
        }
    )SHADER";
}

} // namespace gl
} // namespace renderengine
} // namespace android

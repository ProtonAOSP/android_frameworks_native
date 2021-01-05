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
#include "BlurNoise.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <ui/GraphicTypes.h>
#include <cstdint>

#include <utils/Trace.h>

#undef ALOGI
#define ALOGI(...)

// Minimum and maximum sampling offsets for each pass count, determined empirically.
// Too low: bilinear downsampling artifacts
// Too high: diagonal sampling artifacts
static const std::vector<std::tuple<float, float>> kOffsetRanges = {
    {1.00,  2.50}, // pass 1
    {1.25,  4.25}, // pass 2
    {1.50, 11.25}, // pass 3
    {1.75, 18.00}, // pass 4
    {2.00, 20.00}, // pass 5
    /* limited by kMaxPasses */
};

namespace android {
namespace renderengine {
namespace gl {

BlurFilter::BlurFilter(GLESRenderEngine& engine)
      : mEngine(engine),
        mCompositionFbo(engine),
        mDitherFbo(engine),
        mMixProgram(engine),
        mDitherMixProgram(engine),
        mDownsampleProgram(engine),
        mUpsampleProgram(engine) {
    // Create VBO first for usage in shader VAOs
    static constexpr auto size = 2.0f;
    static constexpr auto translation = 1.0f;
    const GLfloat vboData[] = {
        // Position                              // UV
        translation - size, -translation - size, 0.0f, 0.0f - translation,
        translation - size, -translation + size, 0.0f, size - translation,
        translation + size, -translation + size, size, size - translation,
    };
    mMeshBuffer.allocateBuffers(vboData, 12 /* size */);

    mDownsampleProgram.compile(getDownsampleVertShader(), getDownsampleFragShader());
    mDPosLoc = mDownsampleProgram.getAttributeLocation("aPosition");
    mDUvLoc = mDownsampleProgram.getAttributeLocation("aUV");
    mDTextureLoc = mDownsampleProgram.getUniformLocation("uTexture");
    mDOffsetLoc = mDownsampleProgram.getUniformLocation("uOffset");
    mDHalfPixelLoc = mDownsampleProgram.getUniformLocation("uHalfPixel");
    createVertexArray(&mDVertexArray, mDPosLoc, mDUvLoc);

    mUpsampleProgram.compile(getUpsampleVertShader(), getUpsampleFragShader());
    mUPosLoc = mUpsampleProgram.getAttributeLocation("aPosition");
    mUUvLoc = mUpsampleProgram.getAttributeLocation("aUV");
    mUTextureLoc = mUpsampleProgram.getUniformLocation("uTexture");
    mUOffsetLoc = mUpsampleProgram.getUniformLocation("uOffset");
    mUHalfPixelLoc = mUpsampleProgram.getUniformLocation("uHalfPixel");
    createVertexArray(&mUVertexArray, mUPosLoc, mUUvLoc);

    mMixProgram.compile(getMixVertShader(), getMixFragShader());
    mMPosLoc = mMixProgram.getAttributeLocation("aPosition");
    mMUvLoc = mMixProgram.getAttributeLocation("aUV");
    mMCompositionTextureLoc = mMixProgram.getUniformLocation("uCompositionTexture");
    mMBlurredTextureLoc = mMixProgram.getUniformLocation("uBlurredTexture");
    mMDitherTextureLoc = mMixProgram.getUniformLocation("uDitherTexture");
    mMBlurOpacityLoc = mMixProgram.getUniformLocation("uBlurOpacity");
    createVertexArray(&mMVertexArray, mMPosLoc, mMUvLoc);

    mDitherMixProgram.compile(getMixVertShader(), getDitherMixFragShader());
    mDMPosLoc = mDitherMixProgram.getAttributeLocation("aPosition");
    mDMUvLoc = mDitherMixProgram.getAttributeLocation("aUV");
    mDMCompositionTextureLoc = mDitherMixProgram.getUniformLocation("uCompositionTexture");
    mDMBlurredTextureLoc = mDitherMixProgram.getUniformLocation("uBlurredTexture");
    mDMDitherTextureLoc = mDitherMixProgram.getUniformLocation("uDitherTexture");
    mDMBlurOpacityLoc = mDitherMixProgram.getUniformLocation("uBlurOpacity");
    createVertexArray(&mDMVertexArray, mDMPosLoc, mDMUvLoc);

    mDitherFbo.allocateBuffers(16, 16, (void *) kBlurNoisePattern,
                               GL_NEAREST, GL_REPEAT,
                               GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

status_t BlurFilter::prepareBuffers(const DisplaySettings& display) {
    ATRACE_NAME("BlurFilter::prepareBuffers");

    // Source FBO, used for blurring and crossfading at full resolution
    mDisplayWidth = display.physicalDisplay.width();
    mDisplayHeight = display.physicalDisplay.height();
    mCompositionFbo.allocateBuffers(mDisplayWidth, mDisplayHeight);
    if (mCompositionFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
        ALOGE("Invalid composition buffer");
        return mCompositionFbo.getStatus();
    }

    // Only allocate FBO wrapper objects once
    if (mPassFbos.size() == 0) {
        mPassFbos.reserve(kMaxPasses + 1);
        for (auto i = 0; i < kMaxPasses + 1; i++) {
            mPassFbos.push_back(new GLFramebuffer(mEngine));
        }
    }

    // Allocate FBOs for blur passes, using downscaled display size
    const uint32_t sourceFboWidth = floorf(mDisplayWidth * kFboScale);
    const uint32_t sourceFboHeight = floorf(mDisplayHeight * kFboScale);
    for (auto i = 0; i < kMaxPasses + 1; i++) {
        GLFramebuffer* fbo = mPassFbos[i];

        fbo->allocateBuffers(sourceFboWidth >> i, sourceFboHeight >> i, nullptr,
                                GL_LINEAR, GL_CLAMP_TO_EDGE,
                                // 2-10-10-10 reversed is the only 10-bpc format in GLES 3.1
                                GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV);
        if (fbo->getStatus() != GL_FRAMEBUFFER_COMPLETE) {
            ALOGE("Invalid pass buffer");
            return fbo->getStatus();
        }
    }

    // Creating a BlurFilter doesn't necessarily mean that it will be used, so we
    // only check for successful shader compiles here.
    if (!mMixProgram.isValid()) {
        ALOGE("Invalid mix shader");
        return GL_INVALID_OPERATION;
    }
    if (!mDownsampleProgram.isValid()) {
        ALOGE("Invalid downsample shader");
        return GL_INVALID_OPERATION;
    }
    if (!mUpsampleProgram.isValid()) {
        ALOGE("Invalid upsample shader");
        return GL_INVALID_OPERATION;
    }

    return NO_ERROR;
}

status_t BlurFilter::setAsDrawTarget(const DisplaySettings& display, uint32_t radius) {
    ATRACE_NAME("BlurFilter::setAsDrawTarget");

    mDisplayX = display.physicalDisplay.left;
    mDisplayY = display.physicalDisplay.top;

    // Allocating FBOs is expensive, so only reallocate for larger displays.
    // Smaller displays will still work using oversized buffers.
    if (mDisplayWidth < display.physicalDisplay.width() ||
            mDisplayHeight < display.physicalDisplay.height()) {
        status_t status = prepareBuffers(display);
        if (status != NO_ERROR) {
            return status;
        }
    }

    // Approximate Gaussian blur radius
    if (radius != mRadius) {
        mRadius = radius;
        auto [passes, offset] = convertGaussianRadius(radius);
        ALOGI("SARU: ---------------------------------- new radius: %d  : passes=%d offset=%f", radius, passes, offset);
        mPasses = passes;
        mOffset = offset;
    }

    mCompositionFbo.bind();
    glViewport(0, 0, mCompositionFbo.getBufferWidth(), mCompositionFbo.getBufferHeight());
    return NO_ERROR;
}

std::tuple<int32_t, float> BlurFilter::convertGaussianRadius(uint32_t radius) {
    // Test each pass level first
    for (auto i = 0; i < kMaxPasses; i++) {
        auto [minOffset, maxOffset] = kOffsetRanges[i];
        float offset = radius * kFboScale / std::pow(2, i + 1);
        if (offset >= minOffset && offset <= maxOffset) {
            return {i + 1, offset};
        }
    }

    // FIXME: handle minmax properly
    return {1, radius * kFboScale / std::pow(2, 1)};
}

void BlurFilter::createVertexArray(GLuint* vertexArray, GLuint position, GLuint uv) {
    glGenVertexArrays(1, vertexArray);
    glBindVertexArray(*vertexArray);
    mMeshBuffer.bind();

    glEnableVertexAttribArray(position);
    glVertexAttribPointer(position, 2 /* size */, GL_FLOAT, GL_FALSE,
                          4 * sizeof(GLfloat) /* stride */, 0 /* offset */);

    glEnableVertexAttribArray(uv);
    glVertexAttribPointer(uv, 2 /* size */, GL_FLOAT, GL_FALSE,
                          4 * sizeof(GLfloat) /* stride */,
                          (GLvoid*)(2 * sizeof(GLfloat)) /* offset */);

    mMeshBuffer.unbind();
    glBindVertexArray(0);
}

void BlurFilter::drawMesh(GLuint vertexArray) {
    glBindVertexArray(vertexArray);
    glDrawArrays(GL_TRIANGLES, 0 /* first */, 3 /* vertices */);
    glBindVertexArray(0);
}

void BlurFilter::renderPass(GLFramebuffer* read, GLFramebuffer* draw, GLuint halfPixelLoc, GLuint vertexArray) {
    auto targetWidth = draw->getBufferWidth();
    auto targetHeight = draw->getBufferHeight();
    glViewport(0, 0, targetWidth, targetHeight);

    ALOGI("SARU: blur to %dx%d", targetWidth, targetHeight);

    glBindTexture(GL_TEXTURE_2D, read->getTextureName());
    draw->bind();
    glClear(GL_COLOR_BUFFER_BIT);

    // 1/2 pixel offset in texture coordinate (UV) space
    // Note that this is different from NDC!
    glUniform2f(halfPixelLoc, 0.5 / targetWidth * mOffset, 0.5 / targetHeight * mOffset);
    drawMesh(vertexArray);
}

status_t BlurFilter::prepare() {
    ATRACE_NAME("BlurFilter::prepare");

    glActiveTexture(GL_TEXTURE0);

    GLFramebuffer* read = &mCompositionFbo;
    GLFramebuffer* draw = mPassFbos[0];
    {
        ATRACE_NAME("BlurFilter::blitCompositionBuffer");
        read->bindAsReadBuffer();
        draw->bindAsDrawBuffer();
        glBlitFramebuffer(0, 0,
                        read->getBufferWidth(), read->getBufferHeight(),
                        0, 0,
                        draw->getBufferWidth(), draw->getBufferHeight(),
                        GL_COLOR_BUFFER_BIT, GL_LINEAR);
    }

    ALOGI("SARU: prepare - initial dims %dx%d", mPassFbos[0]->getBufferWidth(), mPassFbos[0]->getBufferHeight());

    // Set up downsampling shader
    mDownsampleProgram.useProgram();
    glUniform1i(mDTextureLoc, 0);

    // Downsample
    for (auto i = 0; i < mPasses; i++) {
        ATRACE_NAME("BlurFilter::renderDownsamplePass");

        // Skip FBO 0 to avoid unnecessary blit
        read = mPassFbos[i];
        draw = mPassFbos[i + 1];

        renderPass(read, draw, mDHalfPixelLoc, mDVertexArray);
    }

    // Set up upsampling shader
    mUpsampleProgram.useProgram();
    glUniform1i(mUTextureLoc, 0);

    // Upsample
    for (auto i = 0; i < mPasses; i++) {
        ATRACE_NAME("BlurFilter::renderUpsamplePass");

        // Upsampling uses buffers in the reverse direction
        read = mPassFbos[mPasses - i];
        draw = mPassFbos[mPasses - i - 1];

        renderPass(read, draw, mUHalfPixelLoc, mUVertexArray);
    }

    mLastDrawTarget = draw;
    return NO_ERROR;
}

status_t BlurFilter::render(size_t layers, int currentLayer) {
    ATRACE_NAME("BlurFilter::render");

    // Now let's scale our blur up. It will be interpolated with the larger composited
    // texture for the first frames, to hide downscaling artifacts.
    GLfloat opacity = fmin(1.0, mRadius / kMaxCrossFadeRadius);

    // When doing multiple passes, we cannot try to read mCompositionFbo, given that we'll
    // be writing onto it. Let's disable the crossfade, otherwise we'd need 1 extra frame buffer,
    // as large as the screen size.
    //if (opacity >= 1 || layers > 1) {
    //    mLastDrawTarget->bindAsReadBuffer();
    //    glBlitFramebuffer(0, 0, mLastDrawTarget->getBufferWidth(),
    //                      mLastDrawTarget->getBufferHeight(), mDisplayX, mDisplayY, mDisplayWidth,
    //                      mDisplayHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    //    return NO_ERROR;
    //}

    // Crossfade using mix shader
    if (currentLayer == layers - 1) {
        mDitherMixProgram.useProgram();
        glUniform1f(mDMBlurOpacityLoc, opacity);
        glUniform1i(mDMCompositionTextureLoc, 0);
        glUniform1i(mDMBlurredTextureLoc, 1);
        glUniform1i(mDMDitherTextureLoc, 2);
    } else {
        mMixProgram.useProgram();
        glUniform1f(mMBlurOpacityLoc, opacity);
        glUniform1i(mMCompositionTextureLoc, 0);
        glUniform1i(mMBlurredTextureLoc, 1);
        glUniform1i(mMDitherTextureLoc, 2);
    }
    ALOGI("SARU: layers=%d current=%d dither=%d", (int)layers, currentLayer, currentLayer == layers - 1);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mCompositionFbo.getTextureName());

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mLastDrawTarget->getTextureName());

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mDitherFbo.getTextureName());

    drawMesh(mMVertexArray);

    // Clean up to avoid breaking further composition
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
    mEngine.checkErrors("Mixing blur");
    return NO_ERROR;
}

string BlurFilter::getDownsampleVertShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform vec2 uHalfPixel;

        in vec2 aPosition;
        in highp vec2 aUV;
        out highp vec2 vUV;
        out vec4 vDownTaps[2];

        void main() {
            vUV = aUV;

            vDownTaps[0] = vec4(aUV - uHalfPixel.xy, aUV + uHalfPixel.xy);
            vDownTaps[1] = vec4(aUV + vec2(uHalfPixel.x, -uHalfPixel.y), aUV - vec2(uHalfPixel.x, -uHalfPixel.y));

            gl_Position = vec4(aPosition, 0.0, 1.0);
        }
    )SHADER";
}

string BlurFilter::getDownsampleFragShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uTexture;

        in highp vec2 vUV;
        in vec4 vDownTaps[2];
        out vec4 fragColor;

        void main() {
            vec4 sum = texture(uTexture, vUV) * 4.0;
            sum += texture(uTexture, vDownTaps[0].xy);
            sum += texture(uTexture, vDownTaps[0].zw);
            sum += texture(uTexture, vDownTaps[1].xy);
            sum += texture(uTexture, vDownTaps[1].zw);
            fragColor = sum * 0.125;
        }
    )SHADER";
}

string BlurFilter::getUpsampleVertShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform vec2 uHalfPixel;

        in vec2 aPosition;
        in highp vec2 aUV;
        out highp vec2 vUV;
        out vec4 vUpTaps[4];

        void main() {
            vUV = aUV;

            vUpTaps[0] = vec4(aUV + vec2(-uHalfPixel.x * 2.0, 0.0), aUV + vec2(-uHalfPixel.x, uHalfPixel.y));
            vUpTaps[1] = vec4(aUV + vec2(0.0, uHalfPixel.y * 2.0), aUV + vec2(uHalfPixel.x, uHalfPixel.y));
            vUpTaps[2] = vec4(aUV + vec2(uHalfPixel.x * 2.0, 0.0), aUV + vec2(uHalfPixel.x, -uHalfPixel.y));
            vUpTaps[3] = vec4(aUV + vec2(0.0, -uHalfPixel.y * 2.0), aUV + vec2(-uHalfPixel.x, -uHalfPixel.y));

            gl_Position = vec4(aPosition, 0.0, 1.0);
        }
    )SHADER";
}

string BlurFilter::getUpsampleFragShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uTexture;

        in highp vec2 vUV;
        in vec4 vUpTaps[4];
        out vec4 fragColor;

        void main() {
            vec4 sum = texture(uTexture, vUpTaps[0].xy);
            sum += texture(uTexture, vUpTaps[0].zw) * 2.0;
            sum += texture(uTexture, vUpTaps[1].xy);
            sum += texture(uTexture, vUpTaps[1].zw) * 2.0;
            sum += texture(uTexture, vUpTaps[2].xy);
            sum += texture(uTexture, vUpTaps[2].zw) * 2.0;
            sum += texture(uTexture, vUpTaps[3].xy);
            sum += texture(uTexture, vUpTaps[3].zw) * 2.0;
            fragColor = sum * 0.08333333333333333;
        }
    )SHADER";
}

string BlurFilter::getMixVertShader() const {
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
            fragColor = mix(composition, blurred, 1.0);
        }
    )SHADER";
}

string BlurFilter::getDitherMixFragShader() const {
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
            vec4 blurred = texture(uBlurredTexture, vUV) + texelFetch(uDitherTexture, ivec2(int(mod(gl_FragCoord.x, 16.0)), int(mod(gl_FragCoord.y, 16.0))), 0);
            vec4 composition = texture(uCompositionTexture, vUV);

            fragColor = mix(composition, blurred, 1.0);
        }
    )SHADER";
}

} // namespace gl
} // namespace renderengine
} // namespace android

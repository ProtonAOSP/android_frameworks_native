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

static const std::vector<std::tuple<float, float>> kOffsetRanges = {
    {1.00, 2.50}, // pass 1
    {1.00, 3.00}, // pass 2
    {1.50, 11.25}, // pass 3
    {1.75, 18.00}, // pass 4
    {2.00, 20.00}, // pass 5
};

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
         28, 183, 111,  75, 232,   7, 156, 150,  52, 236, 252, 103, 217, 192, 226, 163, 125,  85, 248, 140, 252,   1, 203, 185,  63,  65,  34, 132, 216, 230, 101,  89, 191,  24, 175,  21, 193,  98, 225, 224,  20, 132,  62,  83,  71, 104, 111, 219,
        129,  66, 162, 227, 127, 243,  38,  22, 190,   8, 106,  69,  67,  60, 171, 122,  16, 143, 195, 241,  15,  46,  79, 101, 183, 180,  55, 161, 163, 207, 209, 138,  92,  50, 236, 114, 153,  68,  59, 141, 188,  31,  88, 145, 146, 247, 217,  43,
        188,  97,  77, 119, 205, 125, 197,  44, 155, 100, 178, 216, 173, 161,   0, 136, 227,  42,  87,  48, 199, 105,  29, 218, 238, 117, 133,  15,  12, 151,  72,  50,   8, 254,  33, 169,  36, 202, 247, 112, 159, 212,   3,  41, 178, 164,  26, 236,
         20, 168,  11,  52, 243, 203,  84, 136,  33, 252,  82, 117,  31, 215, 246, 231,  93,  79,  18, 147, 161, 152, 187, 109, 216, 249,  70, 121, 225, 242,  92, 154,  45, 166, 109,  78, 185, 128, 124, 232, 254, 104, 203,  57,  19,  70, 228,  90,
        221,   4, 172, 138,  73,  50, 211, 194,  99, 148,  14,  63,  60,  34, 181, 186, 123, 129, 204, 199, 235,  39, 104,  30,  56,  63,  18, 139,  88, 175, 199, 210, 202,   9,  75, 231, 127,   7, 156,  57, 179,  53,  98,  85, 197, 174, 122, 134,
        244, 149, 220,  11, 113, 149, 106, 223, 233, 169, 156,  22,   5, 246, 209, 115,  70,  91,  76,   1,  54, 249, 167, 189, 176,  38, 121,  29, 134,  95, 223, 172, 138,  83,  25,  35, 241, 235,   3,  23, 102, 184, 149, 211,  66,  41, 191, 253,
        116,  52,  82, 184,  37, 108,  73, 100, 193,  43,  58, 137, 226, 184, 167, 130, 139,  10, 159, 233, 147,  96, 207, 224, 110,  18, 254,  69, 119,  60, 147, 244, 195,  47, 197,  84, 107,  47, 222, 212, 155, 144, 133, 132, 116,  81,  17,  28,
         30, 250, 166, 154, 164,   5, 208, 206,  73, 237, 129,  44,  91,  27, 249, 198, 108, 106,  19,  51,  36, 219, 151,  76,   2,  80, 159, 192, 222,  14, 233,  56, 112, 180, 141, 164, 160,  92, 239,  66,  32,  97, 194,  67, 208, 229, 231,  40,
         90, 174, 187,  55,  91, 244, 125,   8, 120,  25, 240, 217, 142, 214,  87,  65,  87, 198,  49, 177, 176, 171, 255, 210, 239,  99, 135, 123, 190,  48,  34,  10, 214,  93, 114,  26,   7, 166,  72, 253, 220,  12,  17, 200, 130, 168,  81, 228,
        201,  21,  88, 245, 118,  20,  10, 189, 153, 177,  76,  58, 190, 148,  27, 255,  15, 126,  32,  42,  65, 150, 124,   2,  85,  30,  98, 135, 158, 182,  59,  74, 245, 207, 182, 123, 143, 238, 173, 120,   2,  51,  44, 107, 157, 102, 144,  61,
         64, 131, 179, 218, 224, 141, 158,  64, 205,  99,  45, 174,  79, 169, 227, 117, 229, 158, 215, 195, 241, 103, 219, 115, 202,  62, 232,  12, 204, 145, 246, 242,  37, 165,  43,  89,  78, 126, 192, 222,  61, 251, 179, 185, 201, 131,  39, 102,
        146,  54, 238,  35, 157,  46, 113, 247, 107,  48, 201,   6, 234,  96, 100, 137, 115,  41,  21,  69, 188, 181, 142,  23,  74,   6,  56,  42, 110,  80, 187, 137, 204, 109,  19,  16,  27,  86, 148,  54, 153, 110, 240, 253,   1,   0, 213,  32,
        228, 176, 127, 172,  24,  68,  71, 105, 215, 210,   5,  81,   4, 135, 248, 167,  28, 136,  61, 165,  74, 225, 248, 150, 126,  95, 223, 157, 173, 170, 230, 212, 131,  97, 230,  64, 214, 103, 229, 155, 198,  75, 191,  13, 140,  82,  94, 221,
         22, 237, 168, 124,  78,  17, 196, 146, 194, 251, 181, 119, 151, 239, 165,  40,  55,  13,  95,  84, 211, 243, 186,  94,  16,  23, 196,  53,  49,  29, 140,  77, 105,   6, 160, 213,  68,  31, 163,  37, 171,  39, 114,  71, 186,  94, 116,  86,
        242, 196, 255,  58,  35, 152,  14, 218,  38,  89, 120, 234, 128, 209,  62, 189,  40, 180, 206, 221,  47, 111, 152, 237, 175, 234,   4,  86, 130, 122, 200, 193, 250, 170,  59,  49, 250, 143,   9, 134, 226, 240, 205,  46, 118, 162, 133,  57,
        213, 162, 183, 144,  53,  93, 182,  90, 139, 108,  11, 206,  51,  72,  25,  26, 170, 128,  77, 101, 113, 145,   3, 160, 220, 112,  67,  33,  36, 142, 235, 251,  83, 118,   9, 177,  80, 121, 154,  13, 208,  96, 178, 245, 200,  45,   0,  24,
    };
    mDitherFbo.allocateBuffers(16, 16, (void *) bayerPattern,
                               GL_NEAREST, GL_REPEAT);
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
        allocPasses = 5;
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
    auto [passes, offset] = convertGaussianRadius(radius);
    ALOGI("SARU: ---------------------------------- new radius: %d  : passes=%d offset=%f", radius, passes, offset);
    if (passes == -1) {
        return BAD_VALUE;
    }
    mPasses = passes;
    mOffset = offset;

    mPasses = 3;
    mOffset = 3.25f;

    mCompositionFbo.bind();
    glViewport(0, 0, mCompositionFbo.getBufferWidth(), mCompositionFbo.getBufferHeight());
    return NO_ERROR;
}

std::tuple<int32_t, float> BlurFilter::convertGaussianRadius(uint32_t radius) {
    for (auto i = 0; i < kMaxPasses; i++) {
        auto [minOffset, maxOffset] = kOffsetRanges[i];
        float offset = radius * kFboScale / std::pow(2, i + 1);
        if (offset >= minOffset && offset <= maxOffset) {
            return {i + 1, offset};
        }
    }

    return {1, radius * kFboScale / std::pow(2, 1)};
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
    glUniform1f(mDOffsetLoc, mOffset);

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

        // 1/2 pixel size in NDC
        glUniform2f(mDHalfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
        drawMesh(mDVertexArray);
    }

    // Set up upsampling shader
    mUpsampleProgram.useProgram();
    glUniform1i(mUTextureLoc, 0);
    glUniform1f(mUOffsetLoc, mOffset);

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

            vec3 dither = texture(uDitherTexture, gl_FragCoord.xy / 16.0).rgb / 64.0;
            blurred = vec4(blurred.rgb + dither, 1.0);

            fragColor = mix(composition, blurred, 1.0);
        }
    )SHADER";
}

} // namespace gl
} // namespace renderengine
} // namespace android

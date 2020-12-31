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
        mBlurSourceFbo(engine),
        mBlurFboH(engine),
        mBlurFboV(engine),
        mMixProgram(engine),
        mBlurProgram(engine) {
    // Create VBO first for usage in shader VAOs
    static constexpr auto size = 2.0f;
    static constexpr auto translation = 1.0f;
    // This represents the rectangular display with a single oversized triangle.
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

    mMixProgram.compile(getMixVertexShader(), getMixFragShader());
    mMPosLoc = mMixProgram.getAttributeLocation("aPosition");
    mMUvLoc = mMixProgram.getAttributeLocation("aUV");
    mMCompositionTextureLoc = mMixProgram.getUniformLocation("uCompositionTexture");
    mMBlurredTextureLoc = mMixProgram.getUniformLocation("uBlurredTexture");
    mMDitherTextureLoc = mMixProgram.getUniformLocation("uDitherTexture");
    mMBlurOpacityLoc = mMixProgram.getUniformLocation("uBlurOpacity");
    mMRadiusLoc = mMixProgram.getUniformLocation("uBlurRadius");
    mMSizeLoc = mMixProgram.getUniformLocation("uBlurSize");
    mMDirLoc = mMixProgram.getUniformLocation("uBlurDir");
    createVertexArray(&mMVertexArray, mMPosLoc, mMUvLoc);

    mBlurProgram.compile(getBlurVertexShader(), getBlurFragShader());
    mBPosLoc = mBlurProgram.getAttributeLocation("aPosition");
    mBUvLoc = mBlurProgram.getAttributeLocation("aUV");
    mBTextureLoc = mBlurProgram.getUniformLocation("uTexture");
    mBRadiusLoc = mBlurProgram.getUniformLocation("uBlurRadius");
    mBSizeLoc = mBlurProgram.getUniformLocation("uBlurSize");
    mBDirLoc = mBlurProgram.getUniformLocation("uBlurDir");
    createVertexArray(&mBVertexArray, mBPosLoc, mBUvLoc);

    mDitherFbo.allocateBuffers(64, 64, (void *) kBlurNoisePattern,
                               GL_NEAREST, GL_REPEAT,
                               GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
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

    // Allocate FBOs for blur passes, using downscaled display size
    const uint32_t sourceFboWidth = floorf(mDisplayWidth * kFboScale);
    const uint32_t sourceFboHeight = floorf(mDisplayHeight * kFboScale);
    mBlurSourceFbo.allocateBuffers(sourceFboWidth, sourceFboHeight, nullptr,
                            GL_LINEAR, GL_CLAMP_TO_EDGE,
                            // 2-10-10-10 reversed is the only 10-bpc format in GLES 3.1
                            GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV);
    if (mBlurSourceFbo.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
        ALOGE("Invalid source blur buffer");
        return mBlurSourceFbo.getStatus();
    }
    mBlurFboH.allocateBuffers(sourceFboWidth, sourceFboHeight, nullptr,
                            GL_LINEAR, GL_CLAMP_TO_EDGE,
                            // 2-10-10-10 reversed is the only 10-bpc format in GLES 3.1
                            GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV);
    if (mBlurFboH.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
        ALOGE("Invalid horizontal blur buffer");
        return mBlurFboH.getStatus();
    }
    mBlurFboV.allocateBuffers(sourceFboWidth, sourceFboHeight, nullptr,
                            GL_LINEAR, GL_CLAMP_TO_EDGE,
                            // 2-10-10-10 reversed is the only 10-bpc format in GLES 3.1
                            GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV);
    if (mBlurFboV.getStatus() != GL_FRAMEBUFFER_COMPLETE) {
        ALOGE("Invalid vertical blur buffer");
        return mBlurFboV.getStatus();
    }

    // Creating a BlurFilter doesn't necessarily mean that it will be used, so we
    // only check for successful shader compiles here.
    if (!mMixProgram.isValid()) {
        ALOGE("Invalid mix shader");
        return GL_INVALID_OPERATION;
    }
    if (!mBlurProgram.isValid()) {
        ALOGE("Invalid downsample shader");
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
        mRadius = max(1.0f, ((float) radius) * kFboScale);
        auto [passes, offset] = std::tuple<int32_t, float>(0, 0.0f); //convertGaussianRadius(radius);
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

void BlurFilter::renderPass(GLFramebuffer* read, GLFramebuffer* draw) {
    auto targetWidth = draw->getBufferWidth();
    auto targetHeight = draw->getBufferHeight();
    glViewport(0, 0, targetWidth, targetHeight);

    ALOGI("SARU: blur to %dx%d", targetWidth, targetHeight);

    glBindTexture(GL_TEXTURE_2D, read->getTextureName());
    draw->bind();

    drawMesh(mBVertexArray);
}

status_t BlurFilter::prepare() {
    ATRACE_NAME("BlurFilter::prepare");

    glActiveTexture(GL_TEXTURE0);

    mCompositionFbo.bindAsReadBuffer();
    mBlurSourceFbo.bindAsDrawBuffer();
    glBlitFramebuffer(0, 0,
                      mCompositionFbo.getBufferWidth(), mCompositionFbo.getBufferHeight(),
                      0, 0,
                      mBlurSourceFbo.getBufferWidth(), mBlurSourceFbo.getBufferHeight(),
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    ALOGI("SARU: prepare - initial dims %dx%d", mBlurSourceFbo.getBufferWidth(), mBlurSourceFbo.getBufferHeight());

    ATRACE_NAME("BlurFilter::renderHorizontalPass");
    mBlurProgram.useProgram();
    glUniform1i(mBTextureLoc, 0);
    glUniform2f(mBSizeLoc, mBlurFboH.getBufferWidth(), mBlurFboH.getBufferHeight());
    glUniform1f(mBRadiusLoc, mRadius / 2.0f);
    glUniform1i(mBDirLoc, 0);
    renderPass(&mBlurSourceFbo, &mBlurFboH);

    ATRACE_NAME("BlurFilter::renderVerticalPass");
    glUniform1i(mBDirLoc, 1);
    renderPass(&mBlurFboH, &mBlurFboV);

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
    glBindTexture(GL_TEXTURE_2D, mBlurFboV.getTextureName());
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

string BlurFilter::getBlurVertexShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform float uBlurRadius;
        uniform vec2 uBlurSize;
        uniform int uBlurDir;

        in vec2 aPosition;
        in highp vec2 aUV;
        out highp vec2 vUV;
        flat out vec2 vOffsetScale;
        flat out int vSupport;
        flat out vec2 vGaussCoefficients;

        void calculateGaussCoefficients(float sigma) {
            // Incremental Gaussian Coefficent Calculation (See GPU Gems 3 pp. 877 - 889)
            vGaussCoefficients = vec2(1.0 / (sqrt(2.0 * 3.14159265) * sigma),
                                    exp(-0.5 / (sigma * sigma)));

            // Pre-calculate the coefficient total in the vertex shader so that
            // we can avoid having to do it per-fragment and also avoid division
            // by zero in the degenerate case.
            vec3 gauss_coefficient = vec3(vGaussCoefficients,
                                        vGaussCoefficients.y * vGaussCoefficients.y);
            float gauss_coefficient_total = gauss_coefficient.x;
            for (int i = 1; i <= vSupport; i += 2) {
                gauss_coefficient.xy *= gauss_coefficient.yz;
                float gauss_coefficient_subtotal = gauss_coefficient.x;
                gauss_coefficient.xy *= gauss_coefficient.yz;
                gauss_coefficient_subtotal += gauss_coefficient.x;
                gauss_coefficient_total += 2.0 * gauss_coefficient_subtotal;
            }

            // Scale initial coefficient by total to avoid passing the total separately
            // to the fragment shader.
            vGaussCoefficients.x /= gauss_coefficient_total;
        }

        void main() {
            vUV = aUV;
            gl_Position = vec4(aPosition, 0.0, 1.0);

            // Ensure that the support is an even number of pixels to simplify the
            // fragment shader logic.
            //
            // TODO(pcwalton): Actually make use of this fact and use the texture
            // hardware for linear filtering.
            vSupport = int(ceil(1.5 * uBlurRadius)) * 2;

            if (vSupport > 0) {
                calculateGaussCoefficients(uBlurRadius);
            } else {
                // The gauss function gets NaNs when blur radius is zero.
                vGaussCoefficients = vec2(1.0, 1.0);
            }

            switch (uBlurDir) {
                case 0:
                    vOffsetScale = vec2(1.0 / uBlurSize.x, 0.0);
                    break;
                case 1:
                    vOffsetScale = vec2(0.0, 1.0 / uBlurSize.y);
                    break;
                default:
                    vOffsetScale = vec2(0.0);
            }
        }
    )SHADER";
}

string BlurFilter::getBlurFragShader() const {
    return R"SHADER(
        #version 310 es
        precision mediump float;

        uniform sampler2D uTexture;

        in highp vec2 vUV;
        flat in vec2 vOffsetScale;
        flat in int vSupport;
        flat in vec2 vGaussCoefficients;
        out vec4 fragColor;

        // blur_radius 0 is NOT supported and MUST be caught before.

        // Partially from http://callumhay.blogspot.com/2010/09/gaussian-blur-shader-glsl.html
        void main() {
            vec4 original_color = texture(uTexture, vUV);

            // Incremental Gaussian Coefficent Calculation (See GPU Gems 3 pp. 877 - 889)
            vec3 gauss_coefficient = vec3(vGaussCoefficients,
                                        vGaussCoefficients.y * vGaussCoefficients.y);

            vec4 avg_color = original_color * gauss_coefficient.x;

            // Evaluate two adjacent texels at a time. We can do this because, if c0
            // and c1 are colors of adjacent texels and k0 and k1 are arbitrary
            // factors, this formula:
            //
            //     k0 * c0 + k1 * c1          (Equation 1)
            //
            // is equivalent to:
            //
            //                                 k1
            //     (k0 + k1) * lerp(c0, c1, -------)
            //                              k0 + k1
            //
            // A texture lookup of adjacent texels evaluates this formula:
            //
            //     lerp(c0, c1, t)
            //
            // for some t. So we can let `t = k1/(k0 + k1)` and effectively evaluate
            // Equation 1 with a single texture lookup.

            for (int i = 1; i <= vSupport; i += 2) {
                gauss_coefficient.xy *= gauss_coefficient.yz;

                float gauss_coefficient_subtotal = gauss_coefficient.x;
                gauss_coefficient.xy *= gauss_coefficient.yz;
                gauss_coefficient_subtotal += gauss_coefficient.x;
                float gauss_ratio = gauss_coefficient.x / gauss_coefficient_subtotal;

                vec2 offset = vOffsetScale * (float(i) + gauss_ratio);

                vec2 st0 = vUV - offset;
                vec2 st1 = vUV + offset;
                avg_color += (texture(uTexture, st0) + texture(uTexture, st1)) *
                            gauss_coefficient_subtotal;
            }

            fragColor = avg_color;
        }
    )SHADER";
}

string BlurFilter::getMixVertexShader() const {
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

            // First /64: screen coordinates -> texture coordinates (UV)
            // Second /64: reduce magnitude to make it a dither instead of an overlay (from Bayer 8x8)
            vec3 dither = texture(uDitherTexture, gl_FragCoord.xy / 64.0).rgb / 64.0;
            blurred = vec4(blurred.rgb + dither, 1.0);

            fragColor = mix(composition, blurred, 1.0);
        }
    )SHADER";
}

} // namespace gl
} // namespace renderengine
} // namespace android

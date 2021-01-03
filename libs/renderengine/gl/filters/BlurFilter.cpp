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
        mMixProgram(engine),
        mDownsampleProgram(engine),
        mUpsampleProgram(engine) {
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

    mDitherFbo.allocateBuffers(16, 16, (void *) kBlurNoisePattern,
                               GL_NEAREST, GL_REPEAT,
                               GL_R8, GL_RED, GL_UNSIGNED_BYTE);
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

void BlurFilter::renderPass(GLFramebuffer* read, GLFramebuffer* draw, GLuint halfPixelLoc, GLuint vertexArray) {
    auto targetWidth = draw->getBufferWidth();
    auto targetHeight = draw->getBufferHeight();
    glViewport(0, 0, targetWidth, targetHeight);

    ALOGI("SARU: blur to %dx%d", targetWidth, targetHeight);

    glBindTexture(GL_TEXTURE_2D, read->getTextureName());
    draw->bind();

    // 1/2 pixel offset in texture coordinate (UV) space
    // Note that this is different from NDC!
    glUniform2f(halfPixelLoc, 0.5 / targetWidth, 0.5 / targetHeight);
    drawMesh(vertexArray);
}

status_t BlurFilter::prepare() {
    ATRACE_NAME("BlurFilter::prepare");

    glActiveTexture(GL_TEXTURE0);

    GLFramebuffer* read = &mCompositionFbo;
    GLFramebuffer* draw = mPassFbos[0];
    read->bindAsReadBuffer();
    draw->bindAsDrawBuffer();
    glBlitFramebuffer(0, 0,
                      read->getBufferWidth(), read->getBufferHeight(),
                      0, 0,
                      draw->getBufferWidth(), draw->getBufferHeight(),
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    ALOGI("SARU: prepare - initial dims %dx%d", mPassFbos[0]->getBufferWidth(), mPassFbos[0]->getBufferHeight());

    // Set up downsampling shader
    mDownsampleProgram.useProgram();
    glUniform1i(mDTextureLoc, 0);
    glUniform1f(mDOffsetLoc, mOffset);

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
    glUniform1f(mUOffsetLoc, mOffset);

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

const float kBlueNoise[256] = float[](

    0.43529411764705883, 0.19215686274509805, 0.5568627450980392, 0.6352941176470588, 0.44313725490196076, 0.7647058823529411, 0.2784313725490196, 0.6941176470588235, 0.788235294117647, 0.19607843137254902, 0.592156862745098, 0.3686274509803922, 0.25882352941176473, 0.1450980392156863, 0.3333333333333333, 0.9882352941176471,
    0.09803921568627451, 0.38823529411764707, 0.9372549019607843, 0.8705882352941177, 0.12549019607843137, 0.9803921568627451, 0.5803921568627451, 0.07450980392156863, 0.14901960784313725, 0.41568627450980394, 0.8627450980392157, 0.6666666666666666, 0.7607843137254902, 0.5411764705882353, 0.050980392156862744, 0.6549019607843137,
    0.49019607843137253, 0.6980392156862745, 0.30980392156862746, 0.058823529411764705, 0.2549019607843137, 0.6784313725490196, 0.4823529411764706, 0.3411764705882353, 0.8352941176470589, 0.5137254901960784, 0.9686274509803922, 0.09019607843137255, 0.4549019607843137, 0.21176470588235294, 0.8980392156862745, 0.8313725490196079,
    0.1607843137254902, 0.792156862745098, 0.596078431372549, 0.5176470588235295, 0.7411764705882353, 0.40784313725490196, 0.20784313725490197, 0.9254901960784314, 0.6313725490196078, 0.24313725490196078, 0.00392156862745098, 0.7098039215686275, 0.30196078431372547, 0.9450980392156862, 0.5764705882352941, 0.26666666666666666,
    0.00784313725490196, 0.9568627450980393, 0.2196078431372549, 0.3568627450980392, 0.9019607843137255, 0.0196078431372549, 0.8, 0.10980392156862745, 0.7333333333333333, 0.396078431372549, 0.5647058823529412, 0.807843137254902, 0.12941176470588237, 0.3607843137254902, 0.7450980392156863, 0.4196078431372549,
    0.8745098039215686, 0.6431372549019608, 0.4470588235294118, 0.1411764705882353, 0.8392156862745098, 0.611764705882353, 0.5450980392156862, 0.27450980392156865, 0.9607843137254902, 0.32941176470588235, 0.8862745098039215, 0.18823529411764706, 0.49411764705882355, 0.6196078431372549, 0.06666666666666667, 0.5294117647058824,
    0.3254901960784314, 0.7686274509803922, 0.08235294117647059, 0.996078431372549, 0.2980392156862745, 0.17647058823529413, 0.7019607843137254, 0.45098039215686275, 0.047058823529411764, 0.1568627450980392, 0.6627450980392157, 0.4117647058823529, 0.9921568627450981, 0.6901960784313725, 0.8274509803921568, 0.23137254901960785,
    0.39215686274509803, 0.7058823529411765, 0.5686274509803921, 0.47843137254901963, 0.6745098039215687, 0.3803921568627451, 0.9215686274509803, 0.5058823529411764, 0.8431372549019608, 0.5843137254901961, 0.7803921568627451, 0.03137254901960784, 0.2823529411764706, 0.10196078431372549, 0.9333333333333333, 0.17254901960784313,
    0.9098039215686274, 0.12156862745098039, 0.27058823529411763, 0.043137254901960784, 0.803921568627451, 0.22745098039215686, 0.07058823529411765, 0.7568627450980392, 0.34509803921568627, 0.23529411764705882, 0.4392156862745098, 0.8666666666666667, 0.5490196078431373, 0.33725490196078434, 0.47058823529411764, 0.6,
    0.8156862745098039, 0.5098039215686274, 0.9529411764705882, 0.6274509803921569, 0.8784313725490196, 0.43137254901960786, 0.13333333333333333, 0.9725490196078431, 0.6470588235294118, 0.09411764705882353, 0.9176470588235294, 0.7215686274509804, 0.20392156862745098, 0.7764705882352941, 0.6705882352941176, 0.023529411764705882,
    0.4235294117647059, 0.7372549019607844, 0.2, 0.34901960784313724, 0.5372549019607843, 0.7294117647058823, 0.6039215686274509, 0.3058823529411765, 0.1843137254901961, 0.5254901960784314, 0.3843137254901961, 0.615686274509804, 0.13725490196078433, 0.9764705882352941, 0.37254901960784315, 0.24705882352941178,
    0.06274509803921569, 0.29411764705882354, 0.8588235294117647, 0.15294117647058825, 0.0, 0.2627450980392157, 0.8941176470588236, 0.4745098039215686, 0.7725490196078432, 0.9411764705882353, 0.011764705882352941, 0.2901960784313726, 0.4980392156862745, 0.0784313725490196, 0.8901960784313725, 0.5607843137254902,
    0.9647058823529412, 0.6862745098039216, 0.4666666666666667, 0.7843137254901961, 0.984313725490196, 0.403921568627451, 0.5725490196078431, 0.054901960784313725, 0.8196078431372549, 0.6823529411764706, 0.42745098039215684, 0.8549019607843137, 0.7529411764705882, 0.3215686274509804, 0.796078431372549, 0.6392156862745098,
    0.11372549019607843, 0.36470588235294116, 0.5882352941176471, 0.08627450980392157, 0.6509803921568628, 0.7137254901960784, 0.21568627450980393, 0.11764705882352941, 0.35294117647058826, 0.25098039215686274, 0.16470588235294117, 0.5529411764705883, 0.6588235294117647, 0.2235294117647059, 0.4588235294117647, 0.1803921568627451,
    0.8470588235294118, 0.9137254901960784, 0.23921568627450981, 0.5019607843137255, 0.3176470588235294, 0.9294117647058824, 0.8509803921568627, 0.4627450980392157, 0.6235294117647059, 1.0, 0.7254901960784313, 0.10588235294117647, 0.9490196078431372, 0.4, 0.01568627450980392, 0.5215686274509804,
    0.28627450980392155, 0.7490196078431373, 0.03529411764705882, 0.8235294117647058, 0.16862745098039217, 0.3764705882352941, 0.027450980392156862, 0.5333333333333333, 0.9058823529411765, 0.3137254901960784, 0.0392156862745098, 0.48627450980392156, 0.8823529411764706, 0.8117647058823529, 0.6078431372549019, 0.7176470588235294
    
);

        float getNoise() {
            int x = int(mod(gl_FragCoord.x, 16.0));
            int y = int(mod(gl_FragCoord.y, 16.0));
            return kBlueNoise[x + y * 16];
        }

        void main() {
            vec4 blurred = texture(uBlurredTexture, vUV);
            vec4 composition = texture(uCompositionTexture, vUV);

            // First /64: screen coordinates -> texture coordinates (UV)
            // Second /64: reduce magnitude to make it a dither instead of an overlay (from Bayer 8x8)
            vec3 noise = vec3(getNoise());
            // Normalize to signed [-0.5; 0.5] range and divide down to (+-)1/255
            // This minimizes visible noise as only a 1/255 step is required for 8-bit quantization
            vec3 dither = (noise - 0.5) / 255.0;
            blurred = vec4(blurred.rgb + dither, 1.0);

            fragColor = mix(composition, blurred, 1.0);
        }
    )SHADER";
}

} // namespace gl
} // namespace renderengine
} // namespace android

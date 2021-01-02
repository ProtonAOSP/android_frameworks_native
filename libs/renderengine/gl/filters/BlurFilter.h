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

#pragma once

#include <ui/GraphicTypes.h>
#include "../GLESRenderEngine.h"
#include "../GLFramebuffer.h"
#include "../GLVertexBuffer.h"
#include "GenericProgram.h"

using namespace std;

namespace android {
namespace renderengine {
namespace gl {

/**
 * This is an implementation of dual-filtered Kawase blur, as described in here:
 * https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
 */
class BlurFilter {
public:
    // Downsample FBO to improve performance
    static constexpr float kFboScale = 1.0f;
    // We allocate FBOs for this many passes to avoid the overhead of dynamic allocation.
    // If you change this, be sure to update kOffsetRanges as well.
    static constexpr uint32_t kMaxPasses = 5;
    // To avoid downscaling artifacts, we interpolate the blurred fbo with the full composited
    // image, up to this radius.
    static constexpr float kMaxCrossFadeRadius = 40.0f;

    explicit BlurFilter(GLESRenderEngine& engine);
    virtual ~BlurFilter(){};

    // Set up render targets, redirecting output to offscreen texture.
    status_t setAsDrawTarget(const DisplaySettings&, uint32_t radius);
    // Execute blur passes, rendering to offscreen texture.
    status_t prepare();
    // Render blur to the bound framebuffer (screen).
    status_t render(bool multiPass);

private:
    uint32_t mRadius;
    uint32_t mPasses;
    float mOffset;

    status_t prepareBuffers(const DisplaySettings& display);
    std::tuple<int32_t, float> convertGaussianRadius(uint32_t radius);
    void createVertexArray(GLuint* vertexArray, GLuint position, GLuint uv);
    void drawMesh(GLuint vertexArray);
    void renderPass(GLFramebuffer* read, GLFramebuffer* draw, GLuint halfPixel, GLuint vertexArray);

    string getVertexShader() const;
    string getDownsampleFragShader() const;
    string getUpsampleFragShader() const;
    string getMixFragShader() const;

    GLESRenderEngine& mEngine;
    // Frame buffer holding the composited background.
    GLFramebuffer mCompositionFbo;
    // Frame buffer holding the Bayer dithering matrix.
    GLFramebuffer mDitherFbo;
    // Frame buffers holding the blur passes. (one extra for final upsample to source FBO size)
    std::vector<GLFramebuffer*> mPassFbos;
    // Buffer holding the final blur pass.
    GLFramebuffer* mLastDrawTarget;

    uint32_t mDisplayWidth = 0;
    uint32_t mDisplayHeight = 0;
    uint32_t mDisplayX = 0;
    uint32_t mDisplayY = 0;

    // VBO containing vertex and uv data of a fullscreen triangle.
    GLVertexBuffer mMeshBuffer;
    GLuint mElementBuffer;

    GenericProgram mMixProgram;
    GLuint mMPosLoc;
    GLuint mMUvLoc;
    GLuint mMBlurOpacityLoc;
    GLuint mMCompositionTextureLoc;
    GLuint mMBlurredTextureLoc;
    GLuint mMDitherTextureLoc;
    GLuint mMVertexArray;

    GenericProgram mDownsampleProgram;
    GLuint mDPosLoc;
    GLuint mDUvLoc;
    GLuint mDTextureLoc;
    GLuint mDOffsetLoc;
    GLuint mDHalfPixelLoc;
    GLuint mDVertexArray;

    GenericProgram mUpsampleProgram;
    GLuint mUPosLoc;
    GLuint mUUvLoc;
    GLuint mUTextureLoc;
    GLuint mUOffsetLoc;
    GLuint mUHalfPixelLoc;
    GLuint mUVertexArray;
};

} // namespace gl
} // namespace renderengine
} // namespace android

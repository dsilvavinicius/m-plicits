#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

// Helper function to apply projection matrix and return NDC space coordinates
glm::vec4 applyProjection(const glm::mat4& projection_matrix, const glm::vec4& point) {
    glm::vec4 projected_point = projection_matrix * point;

    // Perspective divide to bring into NDC space
    projected_point /= projected_point.w;

    return projected_point; // In NDC space
}

// Test case for the projection matrix
TEST(ProjectionMatrixTest, PointsInsideCubeProjectCorrectly) {
    // Setup perspective projection with a camera outside the cube
    float screenWidth = 800.0f;
    float screenHeight = 600.0f;
    float fieldOfView = 45.0f; // Standard FOV
    float nearClip = 0.1f;
    float farClip = 100.0f;

    // Calculate the aspect ratio
    float aspectRatio = screenWidth / screenHeight;

    // Create the perspective projection matrix
    glm::mat4 projection_matrix = glm::perspective(glm::radians(fieldOfView), aspectRatio, nearClip, farClip);

    // Position the camera outside the cube (e.g., at z = 3.0) looking towards the origin
    float cameraDistance = 3.0f;

    // Example points in the scene inside the cube [-1, 1]^3
    glm::vec4 points_in_scene[] = {
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),  // Center of the scene
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f),  // Top-right-front corner
        glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f),// Bottom-left-front corner
        glm::vec4(1.0f, -1.0f, 0.0f, 1.0f), // Bottom-right-front corner
        glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f)  // Top-left-front corner
    };

    // Adjust the Z-coordinate to simulate the camera being outside the cube
    for (auto& point : points_in_scene) {
        point.z -= cameraDistance; // Move points away from the camera
    }

    // Loop through points, apply projection, and check if they are within NDC bounds [-1, 1]
    for (auto& point : points_in_scene) {
        glm::vec4 projected_point = applyProjection(projection_matrix, point);

        // Output projected values for debugging
        std::cout << "Projected Point (x, y, z): ("
            << projected_point.x << ", "
            << projected_point.y << ", "
            << projected_point.z << ")" << std::endl;

        // Verify that each coordinate of the projected point is within [-1, 1]
        EXPECT_GE(projected_point.x, -1.0f);
        EXPECT_LE(projected_point.x, 1.0f);
        EXPECT_GE(projected_point.y, -1.0f);
        EXPECT_LE(projected_point.y, 1.0f);
        EXPECT_GE(projected_point.z, -1.0f); // OpenGL uses [-1, 1] for z in NDC space
        EXPECT_LE(projected_point.z, 1.0f);
    }
}

// Helper function to calculate the ray direction from a given pixel
glm::vec3 calculateRayDirection(int i, int j, int screenWidth, int screenHeight, const glm::mat4& invProjMatrix, const glm::mat4& invViewMatrix) {
    // Convert pixel coordinates to normalized device coordinates (NDC)
    float ndcX = (2.0f * i) / screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * j) / screenHeight; // Flip Y for NDC space

    // Create a point on the near clipping plane in NDC space (z = -1 for the near plane)
    glm::vec4 ray_clip = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);

    // Transform from clip space to view space using the inverse projection matrix
    glm::vec4 ray_eye = invProjMatrix * ray_clip;
    ray_eye.z = -1.0f; // Set the forward direction in view space
    ray_eye.w = 0.0f;  // Direction vector in homogeneous coordinates

    // Transform from view space to world space using the inverse view matrix
    glm::vec3 ray_world = glm::normalize(glm::vec3(invViewMatrix * ray_eye));

    return ray_world; // Return the normalized ray direction in world space
}

// Test case for validating ray directions with a translation-only view matrix
TEST(RayGenerationTest, RayDirectionsWithTranslationOnly) {
    // Setup screen dimensions
    int screenWidth = 800;
    int screenHeight = 600;

    // Setup perspective projection
    float fieldOfView = 45.0f; // FOV in degrees
    float nearClip = 0.1f;
    float farClip = 100.0f;
    float aspectRatio = static_cast<float>(screenWidth) / static_cast<float>(screenHeight);
    glm::mat4 projection_matrix = glm::perspective(glm::radians(fieldOfView), aspectRatio, nearClip, farClip);

    // Setup view matrix with translation only (no rotation)
    glm::vec3 camera_position = glm::vec3(2.0f, 1.0f, 5.0f);  // Translate the camera to this position
    glm::mat4 view_matrix = glm::lookAt(camera_position, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)); // Look at the origin

    // Inverse matrices for ray generation
    glm::mat4 invProjMatrix = glm::inverse(projection_matrix);
    glm::mat4 invViewMatrix = glm::inverse(view_matrix);

    // Calculate the expected center ray direction (should point to the origin)
    glm::vec3 expected_center_ray = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - camera_position);

    // Validate ray directions for specific pixels (e.g., center, corners of the screen)
    glm::vec3 center_ray = calculateRayDirection(screenWidth / 2, screenHeight / 2, screenWidth, screenHeight, invProjMatrix, invViewMatrix);
    glm::vec3 top_left_ray = calculateRayDirection(0, 0, screenWidth, screenHeight, invProjMatrix, invViewMatrix);
    glm::vec3 bottom_right_ray = calculateRayDirection(screenWidth - 1, screenHeight - 1, screenWidth, screenHeight, invProjMatrix, invViewMatrix);

    // Output ray directions for debugging purposes
    std::cout << "Center Ray Direction: (" << center_ray.x << ", " << center_ray.y << ", " << center_ray.z << ")" << std::endl;
    std::cout << "Top Left Ray Direction: (" << top_left_ray.x << ", " << top_left_ray.y << ", " << top_left_ray.z << ")" << std::endl;
    std::cout << "Bottom Right Ray Direction: (" << bottom_right_ray.x << ", " << bottom_right_ray.y << ", " << bottom_right_ray.z << ")" << std::endl;

    // Validate that the center ray points roughly toward the origin, aligned with the expected direction
    EXPECT_NEAR(center_ray.x, expected_center_ray.x, 0.01f);
    EXPECT_NEAR(center_ray.y, expected_center_ray.y, 0.01f);
    EXPECT_NEAR(center_ray.z, expected_center_ray.z, 0.01f);

    // Validate that the top-left ray points to the left and upward
    EXPECT_LT(top_left_ray.x, 0.0f);  // Left direction
    EXPECT_GT(top_left_ray.y, 0.0f);  // Upward direction
    EXPECT_LT(top_left_ray.z, -0.5f); // Forward direction (negative Z)

    // Validate that the bottom-right ray points to the right and downward
    EXPECT_GT(bottom_right_ray.x, 0.0f); // Right direction
    EXPECT_LT(bottom_right_ray.y, 0.0f); // Downward direction
    EXPECT_LT(bottom_right_ray.z, -0.5f); // Forward direction (negative Z)
}
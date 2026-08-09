#include <helper_gl.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

class Arcball {
public:
    Arcball(float width, float height, float fov = 45.0f, float nearPlane = 0.00001f, float farPlane = 10000.f) :
        screenWidth(width),
        screenHeight(height),
        fieldOfView(fov),
        nearClip(nearPlane),
        farClip(farPlane)
    {
        updateProjection();
        updateTransform();
    }

    void resize(float width, float height) {
        cout << "resize: " << width << ", " << height << endl << endl;
        
        screenWidth = width;
        screenHeight = height;
        updateProjection();
    }

    void onClick(float mouseX, float mouseY, int button) {
        auto start_vec = mapToSphere(mouseX, mouseY);

        switch (button) {
        case GLUT_LEFT_BUTTON: start[ROTATION] = start_vec; isDragging[ROTATION] = true; break;
        case GLUT_RIGHT_BUTTON: start[PAN] = glm::vec3(mouseX, mouseY, 0.0f); isDragging[PAN] = true; break;
        }
    }

    void onRelease(int button) {
        switch (button) {
        case GLUT_LEFT_BUTTON: prevQuat = quatRotation; isDragging[ROTATION] = false; break;
        case GLUT_RIGHT_BUTTON: isDragging[PAN] = false; break;
        }
    }

    void onZoom(float delta) {
        float translationSpeed = 0.1f;
        translation_vector.z += delta * translationSpeed;

        updateTransform();
    }

    void onDrag(float mouseX, float mouseY) {
        auto current_vec = mapToSphere(mouseX, mouseY);

        if (isDragging[ROTATION]) {
            current[ROTATION] = current_vec;
            quatRotation = prevQuat * glm::quat(start[ROTATION], current_vec);
        }
        if (isDragging[PAN]) {
            float panSpeed = 0.01f;
            translation_vector += glm::vec3((mouseX - start[PAN].x) * panSpeed, (mouseY - start[PAN].y) * panSpeed, 0.0f);
            start[PAN] = glm::vec3(mouseX, mouseY, 0.0f);
        }

        updateTransform();
    }

    void updateTransform() {
        glm::mat4 rotation = glm::toMat4(quatRotation);
        glm::mat4 translation = glm::translate(glm::identity<glm::mat4>(), translation_vector);
        glm::mat4 scaleMat = glm::scale(glm::identity<glm::mat4>(), glm::vec3(scale, scale, scale));

        view_matrix = translation * scaleMat * rotation;
    }

    void updateProjection() {
        float aspectRatio = screenWidth / screenHeight;
        projection_matrix = glm::perspective(glm::radians(fieldOfView), aspectRatio, nearClip, farClip);
    }

    void setPerspective(float fov, float nearPlane, float farPlane) {
        fieldOfView = fov;
        nearClip = nearPlane;
        farClip = farPlane;
        updateProjection();
    }

	float getFov() { return fieldOfView; }
	float getNear() { return nearClip; }
	float getFar() { return farClip; }

    glm::mat4 getViewMatrix() { return view_matrix; }
    glm::mat4 getProjectionMatrix() { return projection_matrix; }
    
    glm::mat4 getInvViewMatrix() { return glm::inverse(getViewMatrix()); }
    glm::mat4 getInvProjectionMatrix() { return glm::inverse(getProjectionMatrix()); }

private:
    glm::vec3 mapToSphere(float x, float y) {
        glm::vec2 point(1.0 - (x * 2.0 / screenWidth), 1.0 - (y * 2.0 / screenHeight));
        float lengthSquared = point.x * point.x + point.y * point.y;

        if (lengthSquared <= 1.0) {
            return glm::vec3(point, sqrt(1.0 - lengthSquared));
        }
        else {
            float length = sqrt(lengthSquared);
            return glm::vec3(point / length, 0.0f);
        }
    }

    static const enum {
        ROTATION = 0,
        PAN,
        ZOOM
    };

    float screenWidth, screenHeight;
    bool isDragging[3] = { false, false, false };
    glm::vec3 start[3] = { glm::zero<glm::vec3>(), glm::zero<glm::vec3>(), glm::zero<glm::vec3>() };
    glm::vec3 current[3] = { glm::zero<glm::vec3>(), glm::zero<glm::vec3>(), glm::zero<glm::vec3>() };

    glm::quat quatRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::quat prevQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    glm::vec3 translation_vector = glm::zero<glm::vec3>();
    float scale = 1.0f;

    glm::mat4 view_matrix = glm::identity<glm::mat4>();
    glm::mat4 projection_matrix = glm::identity<glm::mat4>();

    float fieldOfView;
    float nearClip;
    float farClip;
};
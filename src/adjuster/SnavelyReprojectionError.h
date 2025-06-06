//
// Created by rainbowx on 25-4-15.
//

#ifndef RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H
#define RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/types.hpp>

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,      // 3 f, l1, l2
                    const T* const image,       // 6 r, t
                    const T* const point,       // 3 x, y, z
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(image, point, p);
        // camera[3,4,5] are the translation.
        p[0] += image[3]; p[1] += image[4]; p[2] += image[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[1];
        const T& l2 = camera[2];
        T r2 = xp*xp + yp*yp;
        T distortion = 1.0 + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[0];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 6, 3>(new SnavelyReprojectionError{observed_x, observed_y});
    }

    double observed_x;
    double observed_y;
};

#endif // RECONSTRUCTION_SNAVELY_REPROJECTIONE_RROR_H

import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self.motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt

        self.update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        # Set
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Set
        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]

        # Calculate
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        # Set
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]

        # Calculate
        mean = np.dot(self.motion_mat, mean)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        covariance = np.linalg.multi_dot((self.motion_mat, covariance, self.motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=.0):
        # Set
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]

        # NSA Kalman
        std = [(1 - confidence) * x for x in std]

        # Calculate
        mean = np.dot(self.update_mat, mean)
        innovation_cov = np.diag(np.square(std))
        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T))

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=.0):
        # Project
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        # Calculate 1
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self.update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        # Calculate 2
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements):
        # Project
        mean, covariance = self.project(mean, covariance)

        # Calculate
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)

        # Calculate distance
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha

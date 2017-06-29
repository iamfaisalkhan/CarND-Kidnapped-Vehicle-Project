/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine gen;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 500;
    for (int i = 0; i < num_particles; i++) {

        // Initialize particles
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);

        // Inititalize weights
        weights.push_back(1.0);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    for (int i = 0; i < particles.size(); i++) {
        double x_next = 1.0;
        double y_next = 1.0;
        double theta_next = 1.0;

        if (fabs(yaw_rate) > 0.000001) {
            x_next = particles[i].x + (velocity/yaw_rate *  (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)));
            y_next = particles[i].y + (velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)));
            theta_next = particles[i].theta + yaw_rate * delta_t;
        } else {
            x_next = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y_next = particles[i].y + velocity * delta_t * sin(particles[i].theta);
            theta_next = particles[i].theta;
        }

        particles[i].x = x_next + noise_x(gen);
        particles[i].y = y_next + noise_y(gen);
        particles[i].theta = theta_next + noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0 ; i < observations.size(); i++) {
        std::vector<std::pair<double, int>> distances;

        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            distances.push_back(std::make_pair(distance, predicted[j].id));
        }

        std::sort(distances.begin(), distances.end());
        // printf("OBS %d has %lf to %d\n", i, distances[0].first, distances[0].second);
        observations[i].id = distances[0].second - 1;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0 ; i < particles.size(); i++) {
        double particleX = particles[i].x;
        double particleY = particles[i].y;
        double particleTheta = particles[i].theta;

        std::vector<LandmarkObs> predicted_lm;

        // for each landmark measure how many of the landmarks are within the sensor range of
        // the particles position. 
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double landmarkX = map_landmarks.landmark_list[j].x_f;
            double landmarkY = map_landmarks.landmark_list[j].y_f;

            double d = dist(particleX, particleY, landmarkX, landmarkY);

            if (d <= sensor_range) {
                LandmarkObs pred;
                pred.id = map_landmarks.landmark_list[j].id_i;
                pred.x = map_landmarks.landmark_list[j].x_f;
                pred.y = map_landmarks.landmark_list[j].y_f;
                predicted_lm.push_back(pred);
            }
        }

        std::vector<LandmarkObs> transformed_obs;
        for (int k = 0; k < observations.size(); k++) {
            LandmarkObs trans_obs;
            trans_obs = observations[k];

            trans_obs.x = particleX + (trans_obs.x * cos(particleTheta) - trans_obs.y * sin(particleTheta));
            trans_obs.y = particleY + (trans_obs.x * sin(particleTheta) + trans_obs.y * cos(particleTheta));
            transformed_obs.push_back(trans_obs);
        }

        dataAssociation(predicted_lm, transformed_obs);

        // Update weights. 
        double W = 1.0;
        for (int k = 0; k < transformed_obs.size(); k++) {
            double muX = map_landmarks.landmark_list[transformed_obs[k].id].x_f;
            double muY = map_landmarks.landmark_list[transformed_obs[k].id].y_f;
            double x = transformed_obs[k].x;
            double y = transformed_obs[k].y;
            double sigX = std_landmark[0];
            double sigY = std_landmark[1];

            double w = exp(-((x-muX)*(x-muX)/(2.*sigX*sigX) + (y-muY)*(y-muY)/(2.*sigY*sigY))) 
                         / (2.*M_PI*sigX*sigY);
            if (w < 1e-6)
                w = 1e-6;

            W *= w;
        }

        particles[i].weight = W;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    // setup weights
    weights.clear();
    for (int i=0; i<particles.size(); i++)
        weights.push_back(particles[i].weight);

    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> resampled_particles;
    for (int i = 0; i < particles.size(); i++) {
        resampled_particles.push_back(particles[d(gen)]);
    }

    particles = resampled_particles;

    for (int i = 0; i < particles.size(); i++)
        particles[i].id = i;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

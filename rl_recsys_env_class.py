#Imports
import gym
import pandas as pd
import numpy as np
from gym import spaces

#Loading the data from the CSVs
candidates_data = pd.read_csv('new_candidates_data.csv')
disability_types = pd.read_csv('disability_types.csv')
disability_sub_types = pd.read_csv('disability_sub_types.csv')
courses = pd.read_csv('courses.csv')

#Declaring the constants
num_candidates = len(candidates_data) #Total number of candidates
num_disability_types = len(disability_types) #Total number of disability types
num_disability_subtypes = len(disability_sub_types) #Total number of disability sub-types
num_qualifications = max(list(candidates_data['candidate_educational_qualifications'])) #Total number of different qualifications
tot_courses = len(courses) #Total number of courses

#Environment Class
class CourseRecommendationEnv(gym.Env):
    def __init__(self):
        # Observation space: [Disability Type ID, Disability Subtype ID, Qualification, Course1, Course2 ... Course50]
        self.observation_space = spaces.MultiDiscrete([num_disability_types+1, num_disability_subtypes+1, num_qualifications+1] + [2] * tot_courses)
        # Action space: Recommend a course
        self.action_space = spaces.Discrete(tot_courses)

        # Initializing state
        self.state = None
        self.candidate_data = self.load_candidate_data()
        self.current_candidate_idx = 0

    def load_candidate_data(self):
        # Loading the candidates data
        data = pd.read_csv('new_candidates_data.csv')
        return data

    def reset(self):
        # Resetting the environment to the initial state
        self.current_candidate_idx = 0
        self.state = (list(np.array(self.candidate_data.iloc[self.current_candidate_idx]))[1:])
        return self.state

    def step(self, action):
        # Executing one time step within the environment
        reward = self.calculate_reward(action)
        self.current_candidate_idx += 1

        if self.current_candidate_idx >= len(self.candidate_data):
            done = True
            next_state = None
        else:
            done = False
            self.state = (list(np.array(self.candidate_data.iloc[self.current_candidate_idx]))[1:])
            next_state = self.state

        return next_state, reward, done, {}

    def calculate_reward(self, action):
        
        total = 0
        same_same_but_diffalent = []
        kinda_same_but_diffalent = []
        all_cours = [0]*50
        #Finding candidates with similar profile
        for i in range(num_candidates):
            if (self.candidate_data.iloc[i, 2] == self.state[1]) and (self.candidate_data.iloc[i, 3] == self.state[2]):
                same_same_but_diffalent.append(i)
            elif (self.candidate_data.iloc[i, 1] == self.state[0]) and (self.candidate_data.iloc[i, 3] == self.state[2]):
                kinda_same_but_diffalent.append(i)

        kinda_same_but_diffalent = np.setdiff1d(kinda_same_but_diffalent, same_same_but_diffalent)
        #Calculating Reward
        if len(same_same_but_diffalent) > 0:
            for ind in same_same_but_diffalent:
                if ind != self.current_candidate_idx:
                    if self.state[action+3] == 0 and self.candidate_data.iloc[ind, action+3] == 1:
                        total += 1
                        all_cours[action] += 1
            for ind in kinda_same_but_diffalent:
                if ind != self.current_candidate_idx:
                    if self.state[action+3] == 0 and self.candidate_data.iloc[ind, action+3] == 1:
                        total += 0.5

        total -= 1
                        
        return total
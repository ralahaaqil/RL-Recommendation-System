# RL-Recommendation-System
The environment for a reinforcement learning project that recommends courses to people with disabilities to enhance their employability.

The project invloves the development of an environment in which an agent learns to recommend a course given a person’s disability and academic profile. The input (state space) is a multi-discrete value with each term representing some form of an individuals profile (For example, the candidates disability type, qualifications and so on). The action taken is a discrete one chosen from a set of courses that can be taken up by the candidates.

The agent receives a positive reward when it chooses a course (action) for an individual (state) that other candidates with similar profiles have taken up. This environment and reward function is structured using the OpenAI Gyms framework. This is then trained using a PPO algorithm offered by Stable Baselines 3.

The model provides efficient results showing good trends with the episode reward mean, entropy coefficient and the explored variance with a stabilization in the clip fraction and approximated KL divergence.

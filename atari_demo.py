import ale_py
# os.environ["SDL_VIDEODRIVER"] = "x11"
# os.environ["DISPLAY"] = ":0"

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
import gymnasium as gym

# Define a function to create the environment for human rendering
def make_renderable_env():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout", render_mode ="human")  # Create the Atari environment
    return env

# Create a single renderable environment
renderable_env = make_renderable_env()

# Create a vectorized environment for training
vec_env = make_atari_env("ALE/Breakout-v5", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Train the model using the vectorized environment
model = A2C("CnnPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25_000)

# Play the game using the trained model and render it
obs = renderable_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = renderable_env.step(action)
    renderable_env.render()  # Renders the environment
    if done:
        obs = renderable_env.reset()

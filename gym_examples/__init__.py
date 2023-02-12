import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="gym_examples/a1-v1",
    entry_point="gym_examples.envs:A1Env",
)

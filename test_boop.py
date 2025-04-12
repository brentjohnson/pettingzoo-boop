from pettingzoo.test import api_test, parallel_api_test
from boop_env import env

def test_boop():
    # Test the environment
    api_test(env(), num_cycles=10, verbose_progress=True)

if __name__ == "__main__":
    test_boop() 
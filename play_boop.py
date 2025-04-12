from boop_env import env

# Create the environment
env = env(render_mode="human")

# Reset the environment
observations, infos = env.reset()

# Play a game
while not env.terminations[env.agent_selection]:
    # Get the current player's observation
    observation = env.observe(env.agent_selection)
    
    # Choose an action (0-35 for the 6x6 grid)
    action = env.action_space(env.agent_selection).sample()
    
    # Step the environment
    env.step(action)
    
    # Render the current state
    env.render()
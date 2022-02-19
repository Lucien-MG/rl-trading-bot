from gym.envs.registration import register

register(
    id='stock_exchange_engine_env-v0',
    entry_point='stock_exchange_engine.envs:StockExchangeEngineEnv',
)

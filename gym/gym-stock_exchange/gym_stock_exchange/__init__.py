from gym.envs.registration import register

register(
    id='stock_exchange_env-v0',
    entry_point='gym_stock_exchange.envs:SimulationStockExchangeEnv',
)

register(
    id='relative_stock_exchange_env-v0',
    entry_point='gym_stock_exchange.envs:RelativeSimulationStockExchangeEnv',
)

register(
    id='stock_exchange_api_env-v0',
    entry_point='gym_stock_exchange.envs:StockExchangeApiEnv',
)

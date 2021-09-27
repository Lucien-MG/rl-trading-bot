from gym.envs.registration import register

register(
    id='gym_stock_exchange-v0',
    entry_point='gym_stock_exchange.envs:StockExchangeEnv',
)
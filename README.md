# rl-trading-bot
A trading bot with reinforcement learning techniques.

For now the data are here:
https://drive.google.com/drive/folders/12xF5dcgRS8i2G13M8oprhiSm11u4GHqd?usp=sharing

## How to run the project ?

Create a venv with python 3 (the project has been tested with python3.x with x >= 7)

```
python3 -m venv venv
```

Then activate the env:

```
source venv/bin/activate
```

Install the requirements:

```
pip3 install -r requirements.txt
```

Finally, run the application:

```
python3 rltrade/rltrade_app.py
```

Or use the command line interface:$


```
python rltrade -h
```

To launch a training use:

```
python rltrade --train -c config/rltrade_config.yaml
```

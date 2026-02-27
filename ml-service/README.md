# Portfolio Optimizer ML Service

A machine learning service for portfolio optimization using reinforcement learning agents (PPO and DQN).

## Project Structure

```
ml-service/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── train.py             # Training script
│   ├── predict.py           # Inference logic
│   ├── config.py            # Configuration settings
│   │
│   ├── env/
│   │   └── portfolio_env.py # Custom Gym environment
│   │
│   ├── models/
│   │   ├── ppo_agent.py     # Proximal Policy Optimization agent
│   │   └── dqn_agent.py     # Deep Q-Network agent
│   │
│   └── utils/
│       ├── data_loader.py   # Data loading utilities
│       ├── reward.py        # Reward functions
│       └── metrics.py       # Performance metrics
│
├── requirements.txt
└── README.md
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Service

Start the FastAPI server:
```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Get portfolio predictions
- `POST /train` - Start training

## Training Models

Run the training script:
```bash
python app/train.py
```

## Configuration

Update `app/config.py` to modify:
- Training hyperparameters
- Portfolio settings
- Model paths
- API configuration

## License

MIT

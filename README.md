# Hand Gesture Recognition

Ek machine learning project jo webcam se hand gestures detect karke pehchanta hai.

## Files

- `collect_data.py` : webcam se gesture data collect karta hai (`data/gestures.csv` mein save hota hai)
- `train_model.py` : data se model train karta hai (`model/gesture_model.pkl` mein save hota hai)
- `main.py` : trained model se real-time gesture recognition chalata hai
- `requirements.txt` : zaroori Python libraries

## Setup

1. Repo clone kar:
   git clone https://github.com/aplha-engineer/Hand-Gesture-Recognition.git
2. Libraries install kar:
   pip install -r requirements.txt

## Kaise chalaye

1. Data collect kar: `python collect_data.py`
2. Model train kar: `python train_model.py`
3. Recognition chala: `python main.py`

## Author

aplha-engineer
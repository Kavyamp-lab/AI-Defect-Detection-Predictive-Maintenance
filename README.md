# AI-Based Defect Detection & Predictive Maintenance System

A real-time industrial monitoring system that combines **Computer Vision** and **Deep Learning** to detect product defects and predict machine failure before breakdown.

# Project Overview
This project performs two major tasks:

#Visual Defect Detection
- Uses **CNN (ResNet18)** model
- Detects faulty products from uploaded images
- Helps improve quality control in manufacturing

#Predictive Maintenance
- Uses **LSTM** model
- Predicts machine failure using sensor telemetry data
- Estimates Remaining Useful Life (RUL)
- Prevents unexpected machine breakdown

#Technologies Used
- Defect Detection: CNN (ResNet18) purpose Detect faulty products
- Predictive Maintenance: LSTM  purpose Predict machine failure 
- Dashboard: Streamlit purpose Real-time monitoring 
- Dataset : NASA C-MAPSS purpose Industrial engine sensor data 

#Dashboard Preview
screenshot in Model prediction result

#How It Works

1. User uploads a component image.
2. The CNN model classifies it as Defective or non defective.
3. Live machine telemetry (Temperature & Vibration) is monitored.
4. LSTM model predicts machine health and potential failure.
5. Results are displayed in a Streamlit dashboard.

#Project Structure`
project-folder/
│
├── dashboard.py
├── models/
├── requirements.txt
├── assets/
│   └── dashboard_output.png
└── README.md
```

---

# Installation & Running

Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

#Install dependencies
pip install -r requirements.txt

#Run the application
streamlit run dashboard.py

#Real-World Applications
- Manufacturing quality control
- Automotive production lines
- Industrial machinery monitoring
- Aerospace engine health tracking
- Smart factories (Industry 4.0)

# Future Improvements

- Deploy on cloud (AWS / Azure)
- Add real-time camera integration
- Improve model accuracy with larger datasets
- Add alert notification system

# Author

By Kavya M Patagar
  

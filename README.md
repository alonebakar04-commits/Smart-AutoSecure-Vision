# Smart-AutoSecure-Vision
A real-time, privacy-first AI surveillance dashboard built with Python Flask, OpenCV, and MongoDB. Features multi-camera live streaming, known person management, suspect tracking, and ISO/IEC-compliant privacy controls. Designed for smart homes, campuses, and enterprises. Final Year Project – University of the Punjab, Gujranwala Campus (2025–2026).

# Smart AutoSecure Vision™

![Dashboard Preview](preview/dashboard.png)  
*(Real-time multi-camera surveillance dashboard with suspect tracking)*

**Smart AutoSecure Vision™** is a **real-time, AI-powered intelligent surveillance system** developed as a Final Year Project at the **University of the Punjab, Gujranwala Campus (2025–2026)**. It transforms ordinary cameras into **autonomous security sentinels** capable of live monitoring, person identification, and threat detection — all while prioritizing **privacy and compliance**.

## 🚀 Key Features

- **Real-Time Multi-Camera Dashboard**  
  - One large main live feed + small clickable grid below  
  - Add any connected camera (prevents duplicates)  
  - Click small camera → instantly becomes main view  

- **Live Person Tracking (WhatsApp-style Log)**  
  - Shows name, photo, time, and direction (In/Out)  
  - Color-coded: Green (Known), Red (Unknown/Suspect)  

- **Today's Summary Panel**  
  - Counters: Suspects, Unknown, Known, Total Traffic (auto-calculated)  
  - Starts at zero — updates dynamically  

- **Admin Panel (/admin)**  
  - Add, Edit, Delete known persons  
  - Fields: Photo, Name, Relation, Phone, Address  
  - Auto-generated **unique serial number**  
  - Data stored securely in **MongoDB**  

- **Privacy & Compliance**  
  - Designed with **ISO/IEC 42001** (AI Management) & **ISO/IEC 30137** (Biometrics) standards  
  - Privacy-by-design architecture (ready for face blurring & liveness in future)  

- **Tech Stack**  
  - **Backend**: Python Flask  
  - **Frontend**: HTML, Bootstrap 5, Vanilla JS  
  - **Database**: MongoDB (local or Atlas)  
  - **Streaming**: OpenCV + MJPEG  

## 📸 Screenshots
ljnjknjnjn
![Main Dashboard](preview/main.png)  
*Live multi-camera view with tracking log*

![Admin Panel](preview/admin.png)  
*Manage known persons with full CRUD*

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- MongoDB (local or Atlas)
- Webcam or IP camera

### Steps
1. Clone the repo
   ```bash
   git clone https://github.com/yourusername/Smart-AutoSecure-Vision.git
   cd Smart-AutoSecure-Vision




   👥 Team

Abu Bakar Iqbal (Group Leader) – BCS22039
Rana Atif – BCS22030
Hamza Tariq – BCS22023
Ali Hassan – BCS21249

Supervisor: Professor Muhammad Younas
University of the Punjab, Gujranwala Campus

📄 License
MIT License – Free to use, modify, and distribute.

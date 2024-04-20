# from flask import Flask, request, jsonify
# import cv2
# import numpy as np

# app = Flask(__name__)

# emotion_model = None
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# @app.route('/detect_emotion', methods=['GET'])
# def detect_emotion():
#     global emotion_model
    
#     if emotion_model is None:
#         from keras.models import load_model

#         import os
#         script_dir = os.path.dirname(os.path.abspath(__file__))  # Sửa '__file__' thành __file__
#         model_path = script_dir + "/model_new.h5"
#         emotion_model = load_model(model_path)
#         print(script_dir)

#     cv2.ocl.setUseOpenCL(False)

#     emotion_count = {emotion: 0 for emotion in emotion_dict.values()}
#     total_frames = 0
#     start_time = cv2.getTickCount()

#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in num_faces:
#             cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#             roi_gray_frame = gray_frame[y:y + h, x:x + w]
            
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            
#             emotion_prediction = np.argmax(emotion_model.predict(cropped_img))
#             emotion_count[emotion_dict[emotion_prediction]] += 1

#         cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

#         total_frames += 1
        
#         # Check if 10 seconds have passed
#         elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
#         if elapsed_time > 10:
#             # Calculate the most frequent emotion
#             most_common_emotion = max(emotion_count, key=emotion_count.get)
#             print("Emotions in the last 10 seconds:")
#             for emotion, count in emotion_count.items():
#                 print(f"{emotion}: {count}")
#             print(f"Most common emotion: {most_common_emotion}")
            
#             # Provide advice based on the most common emotion
#             advice = ""
#             if most_common_emotion == "Happy":
#                 advice ="Rất vui khi nghe bạn đang cảm thấy hạnh phúc! Dưới đây là một số món hàng bạn có thể mua để tăng thêm niềm vui: Đĩa game, Vé xem phim"
#             elif most_common_emotion == "Sad":
#                 advice = "Nếu bạn đang cảm thấy buồn, có thể một số món hàng sau sẽ giúp bạn cảm thấy thoải mái hơn: Khóa học trực tuyến, Trà ấm"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây
#             elif most_common_emotion == "Angry":
#                 advice = "Nếu bạn đang cảm thấy tức giận, có thể một số món hàng sau sẽ giúp bạn giảm căng thẳng và làm dịu cảm xúc: Bộ dụng cụ tập thể dục"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây
#             elif most_common_emotion == "Disgusted":
#                 advice = "Nếu bạn đang cảm thấy kinh tởm, có thể một số món hàng sau sẽ giúp bạn cảm thấy thoải mái và làm dịu cảm xúc: Bộ dụng cụ Skin care, Tinh dầu bưởi"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây
#             elif most_common_emotion == "Fearful":
#                 advice = "Nếu bạn đang cảm thấy sợ hãi, có một số món hàng dưới đây có thể giúp bạn cảm thấy an tâm và tạo ra một không gian an toàn: Đèn ngủ, Tai nghe chống ồn"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây
#             elif most_common_emotion == "Neutral":
#                 advice = "Nếu bạn đang cảm thấy trạng thái trung lập, có một số món hàng dưới đây có thể giúp bạn tìm thêm niềm vui hoặc sự kích thích:Đĩa game, Vé xem phim, Khóa học trực tuyến, Trà ấm"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây
#             elif most_common_emotion == "Surprised":
#                 advice = "Nếu bạn đang cảm thấy ngạc nhiên, có một số món hàng dưới đây có thể giúp bạn tiếp tục tận hưởng cảm xúc này: Kính cận, Bộ dụng cụ Skin care"
#                 # Thêm các món hàng và lời khuyên tương ứng ở đây

#             print(advice)
#             return jsonify({"advice": advice})  # Trả về lời khuyên dưới dạng JSON
#             break
        
#         # Wait for the 'q' key press to exit the loop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close windows
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port="5400")


from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

emotion_model = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

@app.route('/detect_emotion', methods=['GET'])
def detect_emotion():
    global emotion_model
    
    if emotion_model is None:
        from keras.models import load_model

        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = script_dir + "/model_new.h5"
        emotion_model = load_model(model_path)
        print(script_dir)

    cv2.ocl.setUseOpenCL(False)

    emotion_count = {emotion: 0 for emotion in emotion_dict.values()}
    total_frames = 0
    start_time = cv2.getTickCount()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            
            emotion_prediction = np.argmax(emotion_model.predict(cropped_img))
            emotion_count[emotion_dict[emotion_prediction]] += 1

        cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

        total_frames += 1
        
        # Check if 10 seconds have passed
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > 10:
            # Calculate the most frequent emotion
            most_common_emotion = max(emotion_count, key=emotion_count.get)
            print("Emotions in the last 10 seconds:")
            for emotion, count in emotion_count.items():
                print(f"{emotion}: {count}")
            print(f"Most common emotion: {most_common_emotion}")
            
            # Provide advice based on the most common emotion
            advice = ""
            if most_common_emotion == "Happy":
                advice ="Rất vui khi nghe bạn đang cảm thấy hạnh phúc! Dưới đây là một số món hàng bạn có thể mua để tăng thêm niềm vui: Đĩa game, Vé xem phim"
            elif most_common_emotion == "Sad":
                advice = "Nếu bạn đang cảm thấy buồn, có thể một số món hàng sau sẽ giúp bạn cảm thấy thoải mái hơn: Khóa học trực tuyến, Trà ấm"
                # Thêm các món hàng và lời khuyên tương ứng ở đây
            elif most_common_emotion == "Angry":
                advice = "Nếu bạn đang cảm thấy tức giận, có thể một số món hàng sau sẽ giúp bạn giảm căng thẳng và làm dịu cảm xúc: Bộ dụng cụ tập thể dục"
                # Thêm các món hàng và lời khuyên tương ứng ở đây
            elif most_common_emotion == "Disgusted":
                advice = "Nếu bạn đang cảm thấy kinh tởm, có thể một số món hàng sau sẽ giúp bạn cảm thấy thoải mái và làm dịu cảm xúc: Bộ dụng cụ Skin care, Tinh dầu bưởi"
                # Thêm các món hàng và lời khuyên tương ứng ở đây
            elif most_common_emotion == "Fearful":
                advice = "Nếu bạn đang cảm thấy sợ hãi, có một số món hàng dưới đây có thể giúp bạn cảm thấy an tâm và tạo ra một không gian an toàn: Đèn ngủ, Tai nghe chống ồn"
                # Thêm các món hàng và lời khuyên tương ứng ở đây
            elif most_common_emotion == "Neutral":
                advice = "Nếu bạn đang cảm thấy trạng thái trung lập, có một số món hàng dưới đây có thể giúp bạn tìm thêm niềm vui hoặc sự kích thích: Đĩa game, Vé xem phim, Khóa học trực tuyến, Trà ấm"
                # Thêm các món hàng và lời khuyên tương ứng ở đây
            elif most_common_emotion == "Surprised":
                advice = "Nếu bạn đang cảm thấy ngạc nhiên, có một số món hàng dưới đây có thể giúp bạn tiếp tục tận hưởng cảm xúc này: Kính cận, Bộ dụng cụ Skin care"
                # Thêm các món hàng và lời khuyên tương ứng ở đây

            print(advice)
            return advice  # Trả về lời khuyên dưới dạng văn bản tiếng Việt
            break
        
        # Wait for the 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5400")

import cv2

cap = cv2.VideoCapture(0)

chessboardSize = (8,6)
num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        if not ret:
            print("Fail")
            continue
        print("Success")

        cv2.imwrite('images/img' + str(num) + '.png', img)
        print(f"image {num} saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
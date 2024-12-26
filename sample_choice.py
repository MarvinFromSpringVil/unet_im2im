import os
import cv2 
import shutil 


def main():
    X_ROOT = './IHC'
    Y_ROOT = './DAPI'

    DST_ROOT = './DeepStain'

    for f_name in os.listdir(X_ROOT):
        x_path = os.path.join(X_ROOT, f_name)
        y_path = os.path.join(Y_ROOT, f_name)
        x = cv2.imread(x_path)
        y = cv2.imread(y_path)

        dst_x_path = os.path.join(DST_ROOT, os.path.join('x', f_name))
        dst_y_path = os.path.join(DST_ROOT, os.path.join('y', f_name))

        cv2.imshow('window', x)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break 

        elif key == ord('s'): # save 
            shutil.copy2(x_path, dst_x_path)
            shutil.copy2(y_path, dst_y_path)




if __name__ == '__main__':
    main() 
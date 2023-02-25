import cv2
import numpy as np

# 定义全局变量
drawing = False
brush_size = 100
prev_x, prev_y = None, None
flag = cv2.GC_PR_FGD

def render(img_cut, GC_mask_current):

    bursh_color_list = [(0, 255, 0), (0, 0, 255), (0, 255, 0), (0, 0, 255)]

    brush_mask = np.zeros(img_cut.shape[:2])
    bursh_color = bursh_color_list[flag]
    cv2.circle(brush_mask, (prev_x, prev_y), brush_size, (255,), -1)

    preview_img = img_cut.copy()
    preview_img[np.where(GC_mask_current == cv2.GC_FGD)] = preview_img[np.where(GC_mask_current == cv2.GC_FGD)] * 0.3 + np.array(bursh_color_list[cv2.GC_FGD]) * 0.7
    preview_img[np.where(GC_mask_current == cv2.GC_PR_FGD)] = preview_img[np.where(GC_mask_current == cv2.GC_PR_FGD)] * 0.5 + np.array(bursh_color_list[cv2.GC_PR_FGD]) * 0.5
    preview_img[np.where(GC_mask_current == cv2.GC_BGD)] = preview_img[np.where(GC_mask_current == cv2.GC_BGD)] * 0.3 + np.array(bursh_color_list[cv2.GC_BGD]) * 0.7
    preview_img[np.where(GC_mask_current == cv2.GC_PR_BGD)] = preview_img[np.where(GC_mask_current == cv2.GC_PR_BGD)] * 0.5 + np.array(bursh_color_list[cv2.GC_PR_BGD]) * 0.5

    if (flag == 0) | (flag == 1):
        preview_img[np.where(brush_mask == 255)] = preview_img[np.where(brush_mask == 255)] * 0.3 + np.array(bursh_color_list[flag]) * 0.7
    else:
        preview_img[np.where(brush_mask == 255)] = preview_img[np.where(brush_mask == 255)] * 0.5 + np.array(bursh_color_list[flag]) * 0.5

    return preview_img

def draw_circle(event, x, y, flags, param):
    global prev_x, prev_y, drawing, brush_size, flag

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(GC_mask, (x, y), brush_size, (flag,), -1)
            cv2.line(GC_mask, (prev_x, prev_y), (x, y), (flag,), brush_size)
            cv2.circle(GC_mask_current, (x, y), brush_size, (flag,), -1)
            cv2.line(GC_mask_current, (prev_x, prev_y), (x, y), (flag,), brush_size)
        prev_x, prev_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(GC_mask, (x, y), brush_size, (flag,), -1)
        cv2.circle(GC_mask_current, (x, y), brush_size, (flag,), -1)

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_size += 2
        else:
            brush_size -= 2
        if brush_size < 2:
            brush_size = 2



# 读取图像
img_org = cv2.imread('input.png')
img_cut = img_org.copy()
GC_mask = np.zeros(img_org.shape[:2], dtype=np.uint8)
GC_mask_current = np.ones(img_org.shape[:2], dtype=np.uint8)*4


# 显示图像和交互窗口
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    # 显示图片和预览圆形)
    preview_img = render(img_cut, GC_mask_current)
    cv2.imshow('image', preview_img)

    # 处理键盘事件
    k = cv2.waitKey(1) & 0xFF
    if k == 9:
        # 进行GrabCut算法分割
        cv2.grabCut(img=img_org, mask=GC_mask, rect=None, bgdModel=None, fgdModel=None, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)


        # 创建前景掩码
        fgdMask = np.where((GC_mask == cv2.GC_FGD) | (GC_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # 显示原始图像和抠图结果
        img_cut = cv2.bitwise_and(img_org, img_org, mask=fgdMask)

        preview_img = render(img_cut, GC_mask_current)
        cv2.imshow('image', preview_img)
        GC_mask_current = np.ones(img_org.shape[:2], dtype=np.uint8)*4

    elif k == 27:
        break

    elif (k >= 49) & (k <= 51):
        flag = k - 48
    elif k == 161:
        flag = 0

    elif k == 13:
        cv2.imwrite('output.png', img_cut)
        break
        

# 关闭所有窗口
cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt
import cp_hw2
import cv2  
import os
from skimage import io, feature
from scipy import interpolate, signal

def disp(im):
    plt.imshow(im)
    plt.show()

def saveIm(im, name):
    io.imsave(name + '.jpeg', im, quality=95)

def saveData(name, data, multiple=False):
    # if multiply, saves data as a list
    with open(name+'.npy', 'wb') as f:
        if multiple:
            for item in data: 
                np.save(f, item)
        else:
            np.save(f, data)
    print('Saved to file ' + name + '.npy')

def loadData(name):
    data = None
    with open(name+'.npy', 'rb') as f:
        data = np.load(f)
    return data

def uv_coords(lensletSize):
    maxUV = (lensletSize - 1) / 2
    u = np.arange(lensletSize) - maxUV
    v = np.arange(lensletSize) - maxUV
    return u, v

def roundHalfUp(x):
    if x * 10 % 10 > 5:
        return int(x) + 1
    return int(x)

def run_lightfield(path):
    im = io.imread(path)
    print("Computing Lightfield...")
    lightfield(im)

def test_lightfield(im):
    im = io.imread(im)
    L = np.zeros((16, 16, 400, 700, 3))
    for i in range(3):
        L[:, :, :, :, i] = np.reshape(im[:, :, i], (16, 16, 400, 700))
    test = L[0, 0, :, :, :]
    plt.imshow(test / np.max(test))
    plt.show()
    # saveData('test_lightfield', L)
    return L

def lightfield(im):
    # Key note: s is going down, t is going across
    lensletSize = 16
    row, col, ch = im.shape
    numS = row // lensletSize
    numT = col // lensletSize
    L = np.zeros((lensletSize, lensletSize, numS, numT, ch))
    for s in range(numS):
        rowSpan = range(lensletSize*s, lensletSize*(s+1), 1)
        for t in range(numT):
            colSpan = range(lensletSize*t, lensletSize*(t+1), 1) 
            L[:, :, s, t, :] = im[lensletSize*s:lensletSize*(s+1), lensletSize*t:lensletSize*(t+1) :]
    saveData('chessboard_lightfield', L)
    return L

def make_mosaic(lightfield, saveName, flipped=False):
    # lensletSize = 16
    test = False
    display = False
    L = loadData(lightfield)
    L = L.astype(np.double)
    L = L / np.max(L)
   #  print(L[:, :, 0, 0, :].shape)
    u, v, s, t, ch = L.shape
    if flipped:
        s, t, u, v, ch = L.shape

    mosaicIm = np.zeros((u*s, v*t, ch))
    for i in range(u):
        for j in range(v):
            if flipped:
                # print(L[:, :, i, j :].shape)
                # print(i)
                # print(j)
                # print(L[:, :, i,  :].shape)
                mosaicIm[s*i:s*(i+1), t*j:t*(j+1), :] = L[:, :, i, j, :]
            else:
                mosaicIm[s*i:s*(i+1), t*j:t*(j+1), :] = L[i, j, :, :, :]

    if display:
        plt.imshow(mosaicIm)
        plt.show()

    if not test:
        saveIm(mosaicIm, saveName)


# def zero_depth_hacky(im):
#     lensletSize = 16
#     row, col, ch = im.shape
#     numS = row // lensletSize
#     numT = col // lensletSize
#     L = np.zeros((numS, numT, ch))
#     for s in range(numS):
#         rowSpan = range(lensletSize*s, lensletSize*(s+1), 1)
#         for t in range(numT):
#             colSpan = range(lensletSize*t, lensletSize*(t+1), 1)
#             f = im[lensletSize*s:lensletSize*(s+1), lensletSize*t:lensletSize*(t+1), :]
#             L[s, t, :] = np.mean(f)
#     plt.imshow(L / np.max(L))
#     plt.show()

# def zero_depth(lightfield):
#     L = loadData(lightfield)
#     L = L.astype(np.double)
#     L = L / np.max(L)
#     u, v, s, t, ch = L.shape
#     I = np.zeros((s, t, ch))
#     for channel in range(ch):
#         total = 0
#         for i in range(u):
#             for j in range(v):
#                 total += L[i, j, :, :, channel]
#         I[:, :, channel] = total / 256
#     I = I / np.max(I)
#     plt.imshow(I)
#     plt.show()
#     io.imsave('ground_zero.jpeg', I, quality=95)

def get_focal_stack(lightfield, depthList, saveName, aSize=None):
    display = False
    test = False
    # load in the lightfield stored
    L = loadData(lightfield)
    L = L.astype(np.double)
    L = L / np.max(L)
    # initialize the focal stack
    fStack = np.zeros((400, 700, 3, len(depthList)))
    # find each image
    count = 0
    if aSize is None:
        u, v, s, t, c = L.shape
    else:
        u = aSize
        v = aSize
    for d in depthList:
        print("Refocusing lightfield to depth " + str(d) + " with aperture size ", str(u))
        I = refocus(L, u, v, d)
        fStack[:, :, :, count] = I
        count += 1
        if display:
            plt.imshow(I)
            plt.show()
        if not test and aSize is None:
            saveIm(I, saveName + '_' + str(d))
    if not test:
        saveData(saveName + '_a' + str(u), fStack)

def refocus(L, uLim, vLim, d):
    test = False
    display = True

    u, v, s, t, ch = L.shape
    I = np.zeros((s, t, ch))
    uCoords, vCoords = uv_coords(uLim)
    # print(uCoords)

    sList = np.arange(s)
    tList = np.arange(t)

    for channel in range(ch):
        total = 0
        for row in range(uLim):
            for col in range(vLim):
                f = interpolate.interp2d(tList, sList, L[row, col, :, :, channel])
                newInt = f(tList-d*vCoords[col], sList+d*uCoords[row])
                total += newInt

        I[:, :, channel] = total / 256

    I = I / np.max(I)
    return I 

def srgbL_linear(im):
    C = 0.0404482
    lowInds = np.where(im <= C)
    im[im > C] = ((im[im > C] + 0.055) / 1.055)**2.4
    im[lowInds] /= 12.92
    return im


def all_in_focus(F, sig1, sig2, depthList):
    test = False
    display = False

    kSize = 9
    s, t, ch, d = F.shape

    lumStack = np.zeros((s, t, d))
    for depth in range(d):
        I = cp_hw2.lRGB2XYZ(F[:, :, :, depth])
        lumStack[:, :, depth] = I[:, :, 1] # Y channel

    lowStack = cv2.GaussianBlur(lumStack, (kSize, kSize), sigmaX=sig1)
    highStack = lumStack - lowStack
    sharpness = cv2.GaussianBlur(np.square(highStack), (kSize, kSize), sigmaX=sig2)

    focusIm = np.zeros((s, t, ch))
    depthMap = np.zeros((s, t, ch))
    denom = np.sum(sharpness, axis=2)

    first = True
    for channel in range(ch):
        fNum = 0
        if first:
            depthNum = 0
        for depth in range(d):
            fNum += np.multiply(sharpness[:, :, depth], F[:, :, channel, depth])
            if first:
                depthNum += sharpness[:, :, depth] * depthList[depth]
                # plt.imshow(depthNum)

        first = False
        focusIm[:, :, channel] = np.divide(fNum, denom)
        depthMap[:, :, channel] = np.divide(depthNum, denom)
    depthMap = np.max(depthMap) - depthMap

    if display:
        disp(focusIm / np.max(focusIm))
        disp(depthMap / np.max(depthMap))

    if not test:
        saveIm(focusIm, 'all_focus')
        saveIm(depthMap, 'd_map_focus')

def afi(path, imSet):
    # aSize = 7 in this case
    # 
    # im is size 400, 700, 3, depths
    test = False
    display = True
    first = True
    aSize = len(imSet)
    u = 0
    v = 0
    ch = 0
    dSize = 0
    apCount = 0

    for im in imSet:
        imName = im.split('.')[0]
        aperture = int(im.split('_')[3])
        print(imName)
        # print(aperture)
        im = loadData(path + imName)
        if first:
            u, v, ch, dSize = im.shape
            # print(im.shape)
            afiStack = np.zeros((u, v, aSize, dSize, 3))
            first = False
        for row in range(u):
            for col in range(v):
                # print(afiStack[row, col, apCount, :, :])
                afiStack[row, col, apCount, :, :] = np.transpose(im[row, col, :, :]) / aperture
        apCount += 1

    
    uInds = np.random.randint(u, size=4)
    vInds = np.random.randint(v, size=4)
    for i in range(4):
        # print(uInds[i], vInds[i])
        dispIm = afiStack[uInds[i], vInds[i], :, :, :]
        if display:
            disp(dispIm)
        if not test:
            saveIm(dispIm, 'afi_slice_' + str(uInds[i]) + '_' + str(vInds[i]))

    saveName = 'afi_stack'
    if not test:
        saveData(saveName, afiStack)

def afi_depth_map(afiStack):
    display = False
    test = False

    depthList = np.arange(0, 2, step=0.2)
    depthList = np.around(depthList, decimals=1)
    u, v, ap, foc, ch = afiStack.shape
    depthMap = np.zeros((u, v, ch))

    for row in range(u):
        for col in range(v):
            afi = afiStack[row, col, :, :, :]
            varList = np.var(afi[:, :, 0], axis=0) + np.var(afi[:, :, 1], axis=0) + np.var(afi[:, :, 2], axis=0)
            # print(varList)
            depth = np.argmin(varList)
            depthMap[row, col, :] = np.dstack((depth, depth, depth))
        # print(row)

    dispIm = depthMap
    if display:
        disp(dispIm / np.max(dispIm))
    if not test:
        saveIm(dispIm, 'afi_depth_map')


def lightfield_wrapper():
    path = '../data/chessboard_lightfield.png'
    run_lightfield(path)

def mosaic_wrapper():
    path = 'chessboard_lightfield'
    make_mosaic(path)

def afi_wrapper():
    path = 'stack_data/'
    imSet = os.listdir(path)
    realSet = imSet[5:]
    realSet.extend(imSet[:5])
    print(realSet)
    afi(path, realSet)

def afi_dmap_wrapper():
    afiStack = loadData('afi_stack')
    afi_depth_map(afiStack)

def f_d_wrapper():
    lightfield = 'chessboard_lightfield'
    depthList = np.arange(0, 2, step=0.2)
    depthList = np.around(depthList, decimals=1)
    # print(depthList)
    saveName = 'f_d_stack'
    get_focal_stack(lightfield, depthList, saveName)

def c_s_wrapper():
    lightfield = 'chessboard_lightfield'
    depthList = np.arange(0, 1.6, step=0.1)
    depthList = np.around(depthList, decimals=1)
    aList = np.around(np.arange(2, 15))
    # print(depthList)
    for a in aList:
        saveName = 'stack_data/c_s_stack_' + str(a)
        print("Getting stack for size " + str(a))
        get_focal_stack(lightfield, depthList, saveName, aSize=a)

def all_focus_wrapper():
    fStack = 'f_d_stack'
    F = loadData(fStack)
    F = F.astype(np.double)
    F = F / np.max(F)
    # F = srgbL_linear(F)

    sig1 = 0.5
    sig2 = 4

    depthList = np.arange(0, 2, step=0.2)
    depthList = np.around(depthList, decimals=1)

    all_in_focus(F, sig1, sig2, depthList)

def read_vid_stack():
    cap = cv2.VideoCapture('vid.avi')
    vidStack = np.zeros((720, 1280, 3, 61))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        vidStack[:, :, :, count] = frame

        # cv2.imshow('frame', frame)
        
        # input('hi')
        if count == 60:
            break
        count += 1
    saveData('vid_stack', vidStack)

    cap.release()
    cv2.destroyAllWindows()

def examine_video():
    path = 'vid_stack'
    vidStack = loadData(path)
    # x, y, ch, frames = vidStack.shape
    disp(vidStack[:, :, :, 30] / np.max(vidStack[:, :, :, 30]))

def find_shifts(y1, y2, x1, x2, saveName):
    path = 'vid_stack'
    vidStack = loadData(path)
    template = vidStack[y1:y2, x1:x2, :, 30]
    template = template.astype(np.uint8)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # disp(template / np.max(template))
    x, y, ch, frames = vidStack.shape
    tSize = y2 - y1
    shiftArray = np.zeros((frames, 2))
    for f in range(frames):
        frame = vidStack[:, :, :, f]
        frame = frame.astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crossCor = feature.match_template(gray, template, pad_input=True)
        shiftArray[f, :] = np.unravel_index(crossCor.argmax(), crossCor.shape)
    saveData(saveName, shiftArray)

# 680, 420 to 730, 470

def interpolate_shifts(shiftPath, realX, realY, saveName):
    display = False
    test = False
    shifts = loadData(shiftPath)
    
    path = 'vid_stack'
    vidStack = loadData(path)
    x, y, ch, frames = vidStack.shape
    xList = np.arange(x)
    yList = np.arange(y)
    finalFrame = np.zeros((x, y, ch))
    for channel in range(ch):
        for f in range(frames):
            frame = vidStack[:, :, channel, f]
            intFunc = interpolate.interp2d(yList, xList, frame)
            shiftX = shifts[f, 0] - realX
            shiftY = shifts[f, 1] - realY
            shiftFrame = intFunc(yList + shiftY, xList + shiftX)
            finalFrame[:, :, channel] += shiftFrame
    finalFrame = np.dstack((finalFrame[:, :, 2], finalFrame[:, :, 1], finalFrame[:, :, 0]))
    if display:
        disp(finalFrame / np.max(finalFrame))
    if not test:
        saveIm(finalFrame, saveName)


def video_refocus_wrapper():
    # template box 
    interpolate_shifts('shift_data', 385, 700, 'kindbox')



def main():
    # 1.1 Initialize lightfield
    # lightfield_wrapper()

    # 1.2 Lightfield Mosaic
    # path = 'chessboard_lightfield'
    # make_mosaic(path, 'mosaic')


    # 1.3 Refocus Images
    # path = 'chessboard_lightfield'
    # f_d_wrapper()

    # 1.4 All-in-focus and Depth Map
    # all_focus_wrapper()

    # 1.5 Confocal Stereo
    # c_s_wrapper()
    # afi_wrapper()
    # afi_dmap_wrapper()


    # 1.5 also, AFI Mosaic 
    # path = 'afi_stack'
    # make_mosaic(path, 'mosaic_afi', flipped=True)

    # 2.1 Read unstructured lightfield
    # read_vid_stack()

    # Don't use
    # examine_video()

    # 2.2 Refocus for the kind box
    # sName = 'kind'
    # find_shifts(335, 435, 650, 750, sName)
    # interpolate_shifts(sName, (335+435)/2,  (650+750)/2, sName)

    # 2.2 Refocus for the lightbulb box
    # sName = 'lightbulb'
    # find_shifts(400, 600, 150, 350, sName)
    # interpolate_shifts(sName, (400 + 600)/2, (150+350)/2, sName)



if __name__ == '__main__':
    main()



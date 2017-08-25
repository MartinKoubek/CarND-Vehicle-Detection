'''
Created on 3. 8. 2017

@author: ppr00076
'''
import matplotlib.pyplot as plt


def imageShow( img1, img2=None, title2="", img3=None, title3="", \
             img4=None, title4="", img5=None, title5="", img6=None, title6=""):

    ax1 = plt.subplot(2,3,1)
    ax2 = plt.subplot(2,3,2)
    ax3 = plt.subplot(2,3,3)
    ax4 = plt.subplot(2,3,4)
    ax5 = plt.subplot(2,3,5)
    ax6 = plt.subplot(2,3,6)
    
    fontsize = 15
    #f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=fontsize)
    if (img2 != None):
        ax2.imshow(img2)
        ax2.set_title(title2, fontsize=fontsize)
    if (img3 != None):
        ax3.imshow(img3)
        ax3.set_title(title3, fontsize=fontsize)
    if (img4 != None):
        ax4.imshow(img4)
        ax4.set_title(title4, fontsize=fontsize)
    if (img5 != None):
        ax5.imshow(img5)
        ax5.set_title(title5, fontsize=fontsize)
    if (img6 != None):
        ax6.imshow(img6)
        ax6.set_title(title6, fontsize=fontsize)  
        
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
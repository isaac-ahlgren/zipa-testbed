
def sigs_algo(x1,window_len=10000, bands=1000):
    FFTs = []
    from scipy.fft import fft, fftfreq, ifft, rfft, irfft

    if window_len == 0:
        window_len = len(x)

    x = np.array(x1.copy())
    #wind = scipy.signal.windows.hann(window_len)
    for i in range(0,len(x),window_len):
        
        if len(x[i:i+window_len]) < window_len:
            #wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
            x[i:i+window_len] = x[i:i+window_len] #* wind
        else:
            x[i:i+window_len] = x[i:i+window_len] #* wind

        FFTs.append(abs(rfft(x[i:i+window_len])))
 
    E = {}
    bands_lst = []
    for i in range(0,len(FFTs)):
        frame = FFTs[i]
        bands_lst.append([ frame[k:k+bands] for k in range(0,len(frame),bands)])
        for j in range(0,len(bands_lst[i])):
            E[(i,j)] = np.sum(bands_lst[i][j])

    bs = ""
    for i in range(1,len(FFTs)):
        for j in range(0,len(bands_lst[i])-1):
            
            if E[(i,j)] -E[(i,j+1)] - (E[(i-1,j)] - E[(i-1,j+1)]) > 0:
                bs+= "1"
            else:
                bs+="0"   
    return bs

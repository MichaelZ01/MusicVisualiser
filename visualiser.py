import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pygame

import constants

# Audio bar object
class AudioBar:
    def __init__(self, x, y, freq, color):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width = constants.DEFAULT_BAR_WIDTH
        self.height = constants.DEFAULT_BAR_HEIGHT
        self.min_height, self.max_height = constants.MIN_HEIGHT, constants.MAX_HEIGHT
        self.min_decibel, self.max_decibel = constants.MIN_DECIBEL, constants.MAX_DECIBEL
        self.decibel_height_ratio = (self.max_height - self.min_height) / (self.max_decibel - self.min_decibel)
    def update(self, dt, decibel):
        height = (decibel * self.decibel_height_ratio) + self.max_height
        speed = (height - self.height) / 0.1
        self.height += speed * dt
        self.height = self.clamp(self.min_height, self.max_height, self.height)

    def getPoints(self):
            return [(self.x, self.y + self.max_height - self.height), (self.x + self.width, self.y + self.max_height - self.height)]

    def clamp(self, min_value, max_value, value):
        if value < min_value:
            return min_value
        elif value > max_value:
            return max_value
        return value

if __name__ == "__main__":

    # Sample file from librosa
    filename = librosa.example('nutcracker')

    # timeSeries: 1-dimensional numpy.ndarray of floating-point values
    # sampleRate: number of samples recorded per second
    timeSeries, sampleRate = librosa.load(filename)

    # matrix of frequencies and time
    # hop_length: number of audio samples between adjacent frames
    # n_fft: number of samples in each frame
    stft = np.abs(librosa.stft(timeSeries, hop_length=512, n_fft=2048*8))
    
    # Convert amplitude to decibels
    D = librosa.amplitude_to_db(stft, ref=np.max)

    # Get frequencies
    frequencies = librosa.core.fft_frequencies(n_fft=2048*8)
    frequenciesIndexRatio = len(frequencies) / frequencies[len(frequencies) - 1]
    frequencies = np.arange(100, 10000, 100)

    # Get time periods
    times = librosa.core.frames_to_time(np.arange(D.shape[1]), sr=sampleRate, hop_length=512, n_fft=2048*8)
    timeIndexRatio = len(times) / times[len(times) - 1]

    # Get beats of the song
    tempo, beat_frames = librosa.beat.beat_track(y=timeSeries, sr=sampleRate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sampleRate)

    # Initialise 
    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode([990, 800])

    # Create bar objects
    x = 10
    bars = []
    for c in frequencies:
        bars.append(AudioBar(x, 200, c, (51, 153, 255)))
        x += 10

    # Start playing music
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play(0)

    # Timer
    second_count = last = pygame.time.get_ticks()

    # One bar is initially filled
    filledBars = 1
    while True:

        # Filled and Unfilled polygons
        polyFilled = []
        polyUnfilled = []

        # Get current time
        t = pygame.time.get_ticks()

        # Time since last tick
        dt = (t - last) / 1000.0

        # Seconds since start
        seconds = (t - second_count) / 1000
        last = t

        # Increment bars filled if a beat occurs
        if seconds >= beat_times[filledBars - 1]:
            if filledBars == 98:
                filledBars = 1
            else:
                filledBars += 1

        # Fill background
        screen.fill((0, 0, 0))

        # Add polygon coordinates
        polyFilled.append([10, 700])
        polyUnfilled.append([10 * filledBars + 10, 700])

        # Add updated bars
        for i, b in enumerate(bars):
            # Update bar heights
            decibel = D[int(b.freq * frequenciesIndexRatio)][int(pygame.mixer.music.get_pos()/1000.0 * timeIndexRatio)]
            b.update(dt, decibel)

            # Add bars to filled or unfilled
            if i < filledBars:
                polyFilled.append(b.getPoints()[0])
                polyFilled.append(b.getPoints()[1])
            else:
                polyUnfilled.append(b.getPoints()[0])
                polyUnfilled.append(b.getPoints()[1])

        # Finish polygon
        polyFilled.append([10 * filledBars + 10, 700])
        polyUnfilled.append([990, 700])

        # Render polygon
        pygame.draw.polygon(screen, [51, 51, 255], polyFilled)
        pygame.draw.polygon(screen, [255, 255, 255], polyUnfilled)
        pygame.display.update()

        # Check user pressed exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()



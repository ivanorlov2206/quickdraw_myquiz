import subprocess

files = ["apple", "banana", "sun", "square", "star", "bowtie", "book", "cup", "door", "eye", "fish", "sword", "mountain", "donut", "ice cream"]

for file in files:
    subprocess.call(['gsutil', '-m', 'cp', "gs://quickdraw_dataset/full/numpy_bitmap/" + file + ".npy", 'data/'])


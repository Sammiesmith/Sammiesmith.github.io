# code/main.py
import small_align
import big_align
import bells_and_whistles

if __name__ == "__main__":
    print("[main] Running small_align... (naive brute force algo)")
    small_align.main()

    print("[main] Running big_align... (image pyramid algo)")
    big_align.main()

    print("[main] Running bells_and_whistles... (bells and whistles algo)")
    bells_and_whistles.main()


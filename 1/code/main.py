# code/main.py
import small_align
import big_align

if __name__ == "__main__":
    print("[main] Running small_align... (naive brute force algo)")
    small_align.main()

    print("[main] Running big_align... (image pyramid + bells and whistles)")
    big_align.main()

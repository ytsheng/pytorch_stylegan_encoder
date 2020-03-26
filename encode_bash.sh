for file in `ls ../aligned_images/`; do
  echo $file
  fn=$(echo "$file" | sed 's/.jpg//g' | sed 's/.png//g')
  python encode_image.py ../aligned_images/$file latents/$fn.npy --save_optimized_image true
  mv optimized.png optimized_images/optimized_$fn.png
done

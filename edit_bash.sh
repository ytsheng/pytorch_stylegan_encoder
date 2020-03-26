types=('eyeglasses' 'gender' 'smile' 'pose' 'age')
for file in `ls latents/`; do
  echo $file
  fn=$(echo "$file" | sed 's/_01.npy//g' | sed '/^[[:space:]]*$/d')
  for t in ${types[@]}; do
    echo $t
    dir=data/ffhq_expl_results/$fn"_"$t"_wp"
    python InterFaceGAN/edit.py -m stylegan_ffhq -b "InterFaceGAN/boundaries/stylegan_ffhq_"$t"_w_boundary.npy" -i latents/$file -o $dir -s WP
    python InterFaceGAN/edit.py -m stylegan_ffhq -b "InterFaceGAN/boundaries/stylegan_ffhq_"$t"_w_boundary.npy" -i latents/$file -o $dir"_end_5" -s WP --end_distance 5 
  done
#  mv optimized.png optimized_images/optimized_$fn.png
done

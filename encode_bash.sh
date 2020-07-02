start=`date +%s`
echo "Start time: "$start
for file in `ls ../celebrities_aligned/`; do
  echo $file
  fn=$(echo "$file" | sed 's/_01.jpg//g' | sed 's/_01.png//g')
  python encode_image.py --image_path ../celebrities_aligned/$file --dlatent_path latents_celebrities/$fn.npy --save_optimized_image true --optimized_image_path optimized_image_celebrities/optimized_$fn.png
done
end=`date +%s`
runtime=$((end-start))
echo "Running time: "$runtime

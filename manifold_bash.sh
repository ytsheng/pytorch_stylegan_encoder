start=`date +%s`
echo "Start time: "$start
for file in `ls optimized_image_celebrities/`; do
  echo $file
  fn=$(echo "$file" | sed 's/optimized_//g' | sed 's/.png//g')
  mkdir celeb_faces/$fn
  cp optimized_image_celebrities/$file celeb_faces/$fn/$file
  cp latents_celebrities/$fn.npy celeb_faces/$fn/$fn.npy
  cp ../celebrities_aligned/$fn"_01.png" celeb_faces/$fn/$fn.png
done
end=`date +%s`
runtime=$((end-start))
echo "Running time: "$runtime

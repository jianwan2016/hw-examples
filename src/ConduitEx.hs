module ConduitEx where

import           Conduit

main :: IO ()
main = do
  -- Pure operations: summing numbers.
  print $ runConduitPure $ yieldMany [1..10 :: Int] .| sumC

  -- Exception safe file access: copy a file.
  writeFile "input.txt" "This is a test." -- create the source file
  runConduitRes $ sourceFileBS "input.txt" .| sinkFile "output.txt" -- actual copying
  readFile "output.txt" >>= putStrLn -- prove that it worked

  -- Perform transformations.
  print $ runConduitPure $ yieldMany [1..10 :: Int] .| mapC(+ 1) .| sinkList

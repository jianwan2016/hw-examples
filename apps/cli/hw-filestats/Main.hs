{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad
import Data.Monoid
import System.Environment

data Average = Average
  { total :: Int
  , count :: Int
  } deriving (Eq, Show)

single :: Int -> Average
single size = Average size 1

instance Monoid Average where
  mempty = Average 0 0
  mappend (Average at ac) (Average bt bc) = Average (at + bt) (ac + bc)

main :: IO ()
main = do
  args :: [String] <- getArgs
  forM_ args $ \(arg :: String) -> do
    contents :: String <- readFile arg
    let contentLines = lines contents
    let bytesPerLineList = (single . length) <$> contentLines
    let bytesPerLine = mconcat bytesPerLineList
    putStrLn $ "## " <> arg
    if count bytesPerLine > 0
      then do
        let averageBytesPerLine = fromIntegral (total bytesPerLine) / fromIntegral (count bytesPerLine) :: Double
        putStrLn $ "Bytes per line: " <> show averageBytesPerLine
      else putStrLn "Empty file"

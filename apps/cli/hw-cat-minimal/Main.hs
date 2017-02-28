{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad
import System.Environment

main :: IO ()
main = do
  args :: [String] <- getArgs
  forM_ args $ \(arg :: String) -> do
    contents :: String <- readFile arg
    putStrLn contents

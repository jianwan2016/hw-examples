{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad
import System.Environment

dos2unix :: String -> String
dos2unix ('\r':'\n':cs) = '\n':dos2unix cs
dos2unix ('\r'     :cs) = '\n':dos2unix cs
dos2unix (c        :cs) = c   :dos2unix cs
dos2unix []             = []

main :: IO ()
main = do
  args :: [String] <- getArgs
  forM_ args $ \(arg :: String) -> do
    contents :: String <- readFile arg
    putStrLn (dos2unix contents)

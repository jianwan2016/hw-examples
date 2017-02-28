{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import System.Environment
import System.Exit

data Opts = Opts
  { delimeter :: Char
  , field     :: Int
  } deriving (Eq, Show)

usage :: IO ()
usage = putStrLn "Usage: hw-cut-minimal -d <delimeter> -f <field>"

parseOpts :: [String] -> Either String Opts
parseOpts = parseOpts' Nothing Nothing

parseOpts' :: Maybe Char -> Maybe Int -> [String] -> Either String Opts
parseOpts' Nothing  maybeF    ("-d":[c]:cs) = parseOpts' (Just c) maybeF cs
parseOpts' (Just _) _         ("-d":  _: _) = Left "Delimeter already specified"
parseOpts' maybeD   Nothing   ("-f": ns:cs) = parseOpts' maybeD (Just (read ns)) cs
parseOpts' _        (Just _)  ("-f":  _: _) = Left "Field already specified"
parseOpts' (Just d) (Just f)  []            = Right Opts { delimeter = d, field = f }
parseOpts' Nothing  _         []            = Left "Delimeter not specified"
parseOpts' _        Nothing   []            = Left "Field not specified"
parseOpts' _        _         cs            = Left ("Unknown arguments: " ++ show cs)

cut :: Char -> Int -> Int -> String -> String
cut _ _ _ []              = []
cut d f _ ('\n':cs)       = '\n':cut d f 1 cs
cut d f i (c:cs) | c == d = cut d f (i + 1) cs
cut d f i (c:cs) | f == i = c:cut d f i cs
cut d f i (_:cs)          = cut d f i cs

main :: IO ()
main = do
  args :: [String] <- getArgs
  case parseOpts args of
    Left msg -> do
      putStrLn ("Error: " ++ msg)
      putStrLn ""
      usage
      exitWith (ExitFailure 1)
    Right opts -> do
      contents <- getContents
      putStrLn (cut (delimeter opts) (field opts) 1 contents)

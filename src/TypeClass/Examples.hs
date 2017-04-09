module TypeClass.Examples where

import           Data.Semigroup

examples :: IO ()
examples = do
  let _ = () <> ()
  let _ = 2 <> 3 :: Sum Int
  let _ = 2 <> 3 :: Product Int
  let _ = [1, 2] <> [3, 4] :: [Int]
  let _ = (2 <> (3 <> 4) :: Sum Int) == ((2 <> 3) <> 4 :: Sum Int) -- Associative law
  let _ = "Hello" <> " " <> "World" :: String
  let _ = mempty :: ()
  let _ = 2 `mappend` 3 :: Sum Int
  let _ = 2 `mappend` 3 :: Product Int
  let _ = [1, 2] `mappend` [3, 4] :: [Int]
  let _ = "Hello" `mappend` " " `mappend` "World" :: String
  let _ = 1 + 2 :: Int
  let _ = (+) 1 2 :: Int
  let _ = (+1) <$> Just 2 :: Maybe Int
  let _ = ("Hi " ++) <$> Just "Everyone" :: Maybe String
  let _ = (++) <$> Just "Hi " <*> Just "Everyone" :: Maybe String
  let _ = (+)
  return ()

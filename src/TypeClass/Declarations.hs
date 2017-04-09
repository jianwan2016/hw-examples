{-# LANGUAGE NoImplicitPrelude #-}

module TypeClass.Declarations where

class Semigroup a where
  (<>) :: a -> a -> a

  -- Associativity:         x <> (y <> z) === (x <> y) <> z

class Monoid a where
  mempty  :: a
  mappend :: a -> a -> a

  -- Identity:                  mempty `mappend` x === x
  -- Right-identity:            x `mappend` mempty === x
  -- Associativity:    x `mappend` (y `mappend` z) === (x `mappend` y) `mappend` z

class Functor f where
  fmap :: (a -> b) -> f a -> f b

  -- Functor-identity:     fmap id x === id x
  -- Fusion:            fmap (g . f) === fmap g . fmap f

class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

  -- Identity:                           pure id <*> v === v
  -- Homomorphism:                   pure f <*> pure x === pure (f x)
  -- Interchange:                         u <*> pure y === pure ($ y) <*> u
  -- Composition:           pure (.) <*> u <*> v <*> w === u <*> (v <*> w)

class Applicative m => Monad m where
  -- | Sequentially compose two actions, passing any value produced
  -- by the first as an argument to the second.
  (>>=) :: m a -> (a -> m b) -> m b

  return :: a -> m a
  return = pure

  (>=>) :: (a -> m b) -> (b -> m c) -> a -> m c

  -- Left identity:	   return a >>= f === f a
  -- Right identity:     m >>= return === m
  -- Associativity:   (m >>= f) >>= g === m >>= (\x -> f x >>= g)

{-# LANGUAGE NoImplicitPrelude #-}

module TypeClasses.Declarations where

class Semigroup a where
  (<>) :: a -> a -> a

class Monoid a where
  mempty  :: a
  mappend :: a -> a -> a

class Functor f where
  fmap :: (a -> b) -> f a -> f b

class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

class Applicative m => Monad m where
  -- | Sequentially compose two actions, passing any value produced
  -- by the first as an argument to the second.
  (>>=) :: m a -> (a -> m b) -> m b

  --  * @mappend mempty x = x@
  --
  --  * @mappend x mempty = x@
  --
  --  * @mappend x (mappend y z) = mappend (mappend x y) z@
  --
  --  * @mconcat = 'foldr' mappend mempty

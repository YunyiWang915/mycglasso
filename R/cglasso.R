#' Soft Thresholding Function
#'
#' This function performs soft thresholding.
#'
#' @param x Input vector.
#' @param lambda Thresholding parameter.
#' @return Soft thresholded vector.
softThresh <- function(x, lambda) {
  sign(x) * pmax(0, abs(x) - lambda)
}


#' Coordinate Gradient Descent with Lasso Penalty
#'
#' This function implements the coordinate gradient descent algorithm with lasso penalty.
#'
#' @param X The design matrix.
#' @param b The response vector.
#' @param lambda The regularization parameter.
#' @param tol Tolerance for convergence.
#' @param maxit Maximum number of iterations.
#' @return The estimated coefficient vector.
#' @export
#' @examples
#' # example code
#' rho = 0.9
#' n = 200
#' p = 10

#' Sigma <- matrix(0, nrow = p, ncol = p)
#' for (i in 1:p) {
#'  for (j in 1:p) {
#'     Sigma[i,j] = rho^abs(i-j)
#'   }
#' }
#
#' library(MASS)
#' X <- mvrnorm(n = n, mu = rep(0,p), Sigma = Sigma)
#' b = c(1, rep(0, p-1))
#' lambda <- 0.5
#' cg_cd(X, b, lambda, tol = 1e-4, maxit = 1000)


cg_cd <- function(X, b, lambda, tol = 1e-4, maxit = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  u <- numeric(p)
  A <- crossprod(X) / n + 0.01 * diag(p)
  obj <- numeric(maxit + 1)
  u_list <- vector("list", maxit + 1)
  u_list[[1]] <- u

  for (i in 1:maxit) {
    for (j in 1:p) {
      residual <- b[j] - crossprod(A[j, -j], u[-j])
      u[j] <- softThresh(residual, lambda) / A[j, j]
    }
    u_list[[i + 1]] <- u

    obj[i] <- 0.5 * crossprod(u, crossprod(A, u)) - crossprod(u, b) + lambda * sum(abs(u))
    if (max(abs(u_list[[i + 1]] - u_list[[i]])) < tol) {
      break
    }
  }
  return(u)
}


#' ADMM Algorithm for Lasso
#'
#' This function implements the Alternating Direction Method of Multipliers (ADMM) algorithm for solving the Lasso problem.
#' @param X The design matrix.
#' @param b The response vector.
#' @param lambda The regularization parameter.
#' @param tol Tolerance for convergence.
#' @param maxit Maximum number of iterations.
#' @return The estimated coefficient vector.
#' @export
#' @examples
#' # example code
#' rho = 0.9
#' n = 200
#' p = 10

#' Sigma <- matrix(0, nrow = p, ncol = p)
#' for (i in 1:p) {
#'  for (j in 1:p) {
#'     Sigma[i,j] = rho^abs(i-j)
#'   }
#' }
#
#' library(MASS)
#' X <- mvrnorm(n = n, mu = rep(0,p), Sigma = Sigma)
#' b = c(1, rep(0, p-1))
#' lambda <- 0.5
#' cg_admm(X, b, lambda, tol = 1e-4, maxit = 1000)

cg_admm <- function(X, b, lambda, tol = 1e-4, maxit = 1000){
  n <- nrow(X)
  p <- ncol(X)
  gamma <- rep(0, p)
  rho <- 4

  u0 <- u <- z0 <- z<- rep(0,p)
  A <- (1/n) * t(X) %*% X + 0.01 * diag(p)
  Sinv <- solve(A + rho * diag(rep(1,p)))

  for (it in 1:maxit) {
    u <- Sinv %*% (rho * z + b - gamma)

    z <- softThresh(u + gamma/rho, lambda/rho)

    gamma <- gamma + rho*(u - z)

    change <- max(c(base::norm(u - u0, "F"), base::norm(z - z0, "F")))
    if (change < tol || it > maxit) {
      break
    }
    u0 <-  u
    z0 <-  z
  }
  return(u)
}

load_result <-
  function(exp_name,
           param_codes,
           run_codes,
           file_names,
           skip = 0,
           n_max = Inf) {
    runtmp <- NULL
    tmp <- NULL
    num_files <- length(file_names)
    result <- NULL
    for (param in param_codes) {
      for (run in run_codes) {
        runtmp <- read_csv(
          paste(
            data_path,
            exp_name,
            paste("param", param, sep = ""),
            paste("run", run, sep = ""),
            paste(file_names[1], "csv", sep = "."),
            sep = "/"
          ),
          skip = skip,
          n_max = n_max,
          show_col_types = FALSE
        )
        for (name in file_names[2:num_files]) {
          file_name <- paste(
            data_path,
            exp_name,
            paste("param", param, sep = ""),
            paste("run", run, sep = ""),
            paste(name, "csv", sep = "."),
            sep = "/"
          )
          if (file.exists(file_name)) {
            tmp <-
              read_csv(
                file_name,
                skip = skip,
                n_max = n_max,
                show_col_types = FALSE
              )
            runtmp[[name]] <- tmp[[name]]
          } else {
            warning(paste(file_name, " does not exist (filling with 0).\n", sep = ""))
            runtmp[[name]] <- 0
          }
        }
        runtmp$run <- run
        runtmp$param <- param
        runtmp$t <- NULL  # No need to keep the wall time
        result <- bind_rows(result, runtmp)
      }
    }
    result
  }

load_result_np <- function(exp_name,
                           run_codes,
                           file_names,
                           skip = 0,
                           n_max = Inf) {
  runtmp <- NULL
  tmp <- NULL
  num_files <- length(file_names)
  result <- NULL
  
  for (run in run_codes) {
    runtmp <- read_csv(
      paste(
        data_path,
        exp_name,
        paste("run", run, sep = ""),
        paste(file_names[1], "csv", sep = "."),
        sep = "/"
      ),
      skip = skip,
      n_max = n_max,
      # show_col_types = FALSE
    )
    for (name in file_names[2:num_files]) {
      file_name <- paste(
        data_path,
        exp_name,
        paste("run", run, sep = ""),
        paste(name, "csv", sep = "."),
        sep = "/"
      )
      if (file.exists(file_name)) {
        tmp <-
          read_csv(
            file_name,
            skip = skip,
            n_max = n_max,
            # show_col_types = FALSE
          )
        runtmp[[name]] <- tmp[[name]]
      } else {
        warning(paste(file_name, " does not exist (filling with 0).\n", sep = ""))
        runtmp[[name]] <- 0
      }
    }
    runtmp$run <- run
    runtmp$t <- NULL  # No need to keep the wall time
    result <- bind_rows(result, runtmp)
  }
  result
}

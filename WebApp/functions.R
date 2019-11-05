

validate_mturk_id <- function(mturk_id) {
  if (nchar(mturk_id) < 12 | nchar(mturk_id) > 14) {
    return(
      list(
        error_message = "Your MTurk ID must be 12-14 characters long",
        result = F
      )
    )
  }

a <- as.numeric(Sys.time())
set.seed(a)
    
  past_ids <- gs_title(x = "epa_results") %>%
    gs_read(ws = 1, cell_cols(x = 1)) %>%
    as.data.frame()
  
  if (nrow(past_ids) > 0) {
    if (mturk_id %in% unique(past_ids$mturk_id)) {
      return(
        list(
          error_message = paste(
            "User", 
            mturk_id, 
            "has already participated."
          ),
          result = F
        )
      )
    }
  }
  
  list(
    modal_error_message = NULL,
    result = T
  )
}

populate_bank <- function(params, phrases) {
  df <- data.frame(
    stringsAsFactors = F, 
    phrase = character(), 
    type = character(),
    used = numeric(),
    time = numeric(),
    rating_value = character()
  )  
  
  if (params$logic == "same_phrase") {
    distinct_phrases <- params$max_ratings / 3
    block_size <- params$same_type_block / 3
  } else {
    distinct_phrases <- params$max_ratings
    block_size <- params$same_type_block
  }
    
  while (nrow(df) < distinct_phrases) {
    type <- sample(x = unique(phrases$type), size = 1)
    
    df <- bind_rows(
      df,
      sample_n(
        tbl = phrases[phrases$type == type, c("phrase", "type")], 
        size = block_size
      )
    )
  }
    
  if (params$logic == "same_phrase") {
    df$phrase_id <- 1:(params$max_ratings / 3)
    df <- bind_rows(df, df, df) %>% arrange(phrase_id)
  }
  
  dimension <- character()
  
  if (params$dimension_order == "Random") {
    if (params$logic == "same_phrase") {
      for (i in 1:distinct_phrases) {
        dimension <- c(
          dimension, 
          sample(
            x = c("evaluative", "potency", "activity"), 
            size = 3
          ) 
        )
      }
    } else {
      for (i in 1:(params$max_ratings / params$same_dimension_n)) {
        dimension <- c(
          dimension, 
          rep(
            x = sample(x = c("evaluative", "potency", "activity"), size = 1), 
            times = params$same_dimension_n
          )
        )
      }
    }
  } else {
    dimension_order <- params$dimension_order %>% 
      toupper() %>%
      strsplit(split = "") %>% 
      unlist()
    
    dimension_order <- c(
      E = "evaluative", 
      P = "potency", 
      A = "activity"
    )[dimension_order] %>%
      unname()
    
    if (params$logic == "same_phrase") {
      dimension <- rep(x = dimension_order, times = distinct_phrases)
    } else {
      dimension_order <- rep_len(
        x = dimension_order, 
        length.out = params$max_ratings / params$same_dimension_n
      )
      
      for (i in dimension_order) {
        dimension <- c(dimension, rep(x = i, times =))
      }
    }
  }
    
  df$dimension <- dimension
  
  df
}

save_results <- function(bank, mturk_id, remote_addr) {
  df <- bank %>%
    filter(used == 1) %>%
    mutate(mturk_id = mturk_id, ip = remote_addr) %>%
    select(mturk_id, ip, phrase, dimension, rating_value, time)
  
  results_sheet <- gs_title(x = "epa_results")
  
  df <- results_sheet %>%
    gs_read(ws = 1) %>%
    as.data.frame() %>%
    mutate(rating_value = as.character(rating_value)) %>%
    bind_rows(df)
  
  write.xlsx2(
    x = df, 
    file = "temp.xlsx", 
    sheetName = "results", 
    col.names = T, 
    row.names = F, 
    append = F
  )
  
  drive_update(
    file = as_id(results_sheet$sheet_key), 
    media = "temp.xlsx"
  )
  
  file.remove("temp.xlsx")
}

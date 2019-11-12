function(input, output, session) {
  # reactive values -----
  logged <- reactiveVal(value = NA)
  bank <- reactiveVal(value = NA)
  counter <- reactiveVal(value = NA)
  timer <- reactiveVal(value = NA)
  
  message <- reactiveVal(
    value = "Welcome to the EPA Rating web app.<br>Please log in to continue."
  )
  
  phrases <- gs_title(x = "epa_phrases") %>%
    gs_read(ws = "phrases") %>%
    as.data.frame()
  
  settings <- gs_title(x = "epa_settings") %>%
    gs_read(ws = 1) %>%
    as.data.frame()
  
  params <- reactiveValues(
    max_ratings = settings[1, 3] %>% as.numeric(),
    ratings_round = settings[2, 3] %>% as.numeric(),
    same_type_block = settings[3, 3] %>% as.numeric(),
    same_dimension_n = settings[4, 3] %>% as.numeric(),
    logic = settings[5, 3],
    dimension_order = settings[6, 3],
    money_per_round = settings[7, 3] %>% as.numeric()
  )
  
  # info -----
  observeEvent(input$info, {
    showModal(
      modalDialog(
        title = NULL,
        footer = NULL,
        size = "s",
        easyClose = T,
        h3("About"),
        hr(),
        "In this study, we will ask you to rate a variety of words.", 
        br(), br(),
        "Please move the slider before submitting your answer",
        hr(),
        modalButton(label = "Got it")
      )
    )
  })
  
  # login panel -----
  output$login_panel <- renderUI({
    if (is.na(logged())) {
      tagList(
        textInput(
          inputId = "mturk_id",
          label = "MTurk ID",
          value = NULL,
        ),
        actionButton(
          inputId = "login",
          label = "Login"
        )
      )
    } else {
      HTML(
        paste0(
          "Logged in as ", 
          strong(logged()),
          " (",
          actionLink(inputId = "logout", label = "Log out"),
          ")"
        )
      )
    }
  })
  
  observeEvent(input$login, ignoreInit = T, {
    req(input$mturk_id)

    withProgress(
      message = "Please wait",
      value = 0.5,
      expr = {
        validation <- validate_mturk_id(mturk_id = input$mturk_id)
      }
    )
    
    if (validation$result) {
      showModal(
        modalDialog(
          title = NULL,
          size = "s",
          footer = NULL,
          easyClose = F,
          h3("About"),
          hr(),
          "In this study, we will ask you to rate a variety of words.",
          br(), br(),
          "Please move the slider before submitting your answer",
          hr(),
          actionButton(inputId = "about_modal", label = "Got it")
        )
      )
    } else {
      message(
        "Welcome to the EPA Rating web app.<br>Please log in to continue."
      )
      
      showModal(
        modalDialog(
          title = NULL,
          footer = NULL,
          size = "s",
          easyClose = T,
          h3("Error"),
          hr(),
          validation$error_message,
          hr(),
          modalButton(label = "Dismiss")
        )
      )
    }
  })
  
  observeEvent(input$about_modal, {
    logged(input$mturk_id)
    counter(1)
    removeModal()
    bank(populate_bank(params = params, phrases = phrases))
  })
  
  observeEvent(input$logout, ignoreInit = T, {
    logged(NA)
    counter(NA)
    bank(NA)
    message("Welcome to the EPA Rating web app.<br>Please log in to continue.")
  })
  
  # message panel -----
  output$message <- renderText({
    req(is.na(logged()))
    message()
  })
  
  # ratings panel -----
  observe({
    req(input$rating != 0)
    enable(id = "submit")
  })
  
  output$ratings_panel <- renderUI({
    req(bank())
    
    if (counter() > params$max_ratings) {
      withProgress(
        value = 0.5, 
        message = "Saving results, please wait",
        expr = save_results(
          bank = bank(), 
          mturk_id = input$mturk_id, 
          remote_addr = input$remote_addr
        )
      )
      
      logged(NA)
      counter(NA)
      bank(NA)
      showModal(
        modalDialog(
          title = NULL,
          footer = NULL,
          size = "s",
          easyClose = T,
          h3("IMPORTANT: Payment"),
          hr(),
          "Thank you for your participation!", 
          br(), br(),
          "Please enter your MTurk ID as your survey code to receive payment.",
          hr(),
          modalButton(label = "Got it")
        )
      )
      message(
        paste0(
          "Have a great day!"
        )
      )
      
      return()
    } 
    
    if (
      (counter() - 1) %in% seq(
        from = params$ratings_round, 
        to = params$max_ratings, 
        by = params$ratings_round
      )
    ) { 
      showModal(
        modalDialog(
          title = NULL, 
          footer = NULL, 
          size = "s", 
          easyClose = F,
          h3("Milestone Reached"),
          hr(),
          "You have reached a milestone.",
          br(), br(),
          "Click \"Continue\" if you would like to do another", 
          params$ratings_round, 
          paste0("phrases for $", params$money_per_round, ","),
          "or click \"Stop\" if you would like to stop now.",
          br(), br(),
          fluidRow(
            column(
              width = 5,
              offset = 1,
              align = "right",
              actionButton(
                inputId = "stop", 
                label = "Stop", 
                width = "100%"
              )
            ),
            column(
              width = 5,
              align = "left",
              actionButton(
                inputId = "continue", 
                label = "Continue", 
                width = "100%"
              )
            )
          )
        )
      )
    }
    
    tl <- tagList(
      fluidRow(
        column(
          width = 12, 
          align = "center", 
          h3(bank()[counter(), "phrase"])
        )
      ),
      br(),
      
      fluidRow(
        column(
          width = 3,
          align = "right",
          scale_anchors[[bank()[counter(), "dimension"]]][[1]] %>% 
            paste(collapse = "<br>") %>%
            HTML()
        ),
        column(
          width = 6, 
          class = "slider-row",
          noUiSliderInput(
            inputId = "rating", 
            label = NULL, 
            min = -4.3, 
            max = 4.3, 
            value = 0, 
            step = 0.1,
            tooltips = F, 
            connect = F,
            pips = list(mode = "values", values = c(-4.3, -3:3, 4.3)),
            behaviour = "snap",
            width = "100%",
            update_on = "change"
          )
        ),
        column(
          width = 3,
          align = "left",
          scale_anchors[[bank()[counter(), "dimension"]]][[2]] %>% 
            paste(collapse = "<br>") %>%
            HTML()
        )
      ),
      
      fluidRow(
        column(
          class = "labels-row", 
          width = 6, 
          offset = 3,
          span("infinitely", class = "slider-labels", style = "left: 0%"),
          span("extremely", class = "slider-labels", style = "left: 14%"),
          span("quite", class = "slider-labels", style = "left: 27.75%"),
          span("slightly", class = "slider-labels", style = "left: 38.6%"),
          span("neutral", class = "slider-labels", style = "left: 50%"),
          span("slightly", class = "slider-labels", style = "left: 61.62791%"),
          span("quite", class = "slider-labels", style = "left: 74.2%"), 
          span("extremely", class = "slider-labels", style = "left: 83.8%"),
          span("infinitely", class = "slider-labels", style = "left: 99.5%")
        )
      ),
      
      br(),
      
      fluidRow(
        column(
          width = 2,
          offset = 4,
          actionButton(
            inputId = "skip", 
            label = "Skip", 
            width = "100%"
          )
        ),
        column(
          width = 2,
          disabled(
            actionButton(
              inputId = "submit", 
              label = "Submit", 
              width = "100%"
            )
          )
        )
      )
    )
    
    timer(Sys.time())
    tl
  })
  
  observeEvent(input$continue, ignoreInit = T, ignoreNULL = T, {
    removeModal()
    timer(Sys.time())
  })
  
  observeEvent(input$stop, ignoreInit = T, ignoreNULL = T, {
    withProgress(
      value = 0.5, 
      message = "Saving results, please wait",
      expr = save_results(
        bank = bank(), 
        mturk_id = input$mturk_id, 
        remote_addr = input$remote_addr
      )
    )
    
    removeModal()
    
    message(
      paste0(
        "You have completed ", 
        counter() - 1, 
        " phrases.<br>Thank you for your participation."
      )
    )
    
    logged(NA)
    counter(NA)
    bank(NA)
  })
  
  observeEvent(input$submit, ignoreInit = T, {
    req(bank())
    
    b <- bank()
    
    b[counter(), "time"] <- difftime(
      time1 = Sys.time(), 
      time2 = timer(), 
      units = "secs"
    ) %>% 
      as.numeric()
    
    b[counter(), "used"] <- 1
    
    b[counter(), "rating_value"] <- input$rating
    
    bank(b)
    
    counter(counter() + 1)
  })
  
  observeEvent(input$skip, ignoreInit = T, {
    req(bank())
    
    b <- bank()
    
    b[counter(), "time"] <- difftime(
      time1 = Sys.time(), 
      time2 = timer(), 
      units = "secs"
    ) %>% 
      as.numeric()
    
    b[counter(), "used"] <- 1
    
    b[counter(), "rating_value"] <- "SKIP"
    
    bank(b)
    
    counter(counter() + 1)
  })
}

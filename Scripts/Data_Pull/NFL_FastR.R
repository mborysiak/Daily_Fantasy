library(nflfastR)
library(arrow)
library(nflreadr)
library(data.table)
library(future)

future::plan("multisession")

#------------------
# Pull in the Play-by-Play Data
#------------------

# define which seasons shall be loaded
season_pull <- 2024
seasons <- season_pull
pbp <- data.table(nflreadr::load_pbp(seasons))

# pull in the roster data
rosters <- data.table(nflreadr::load_rosters(seasons))

# remove full player names and filter down to real players
pbp[, c('receiver_player_name', 'rusher_player_name', 'passer_player_name') := NULL]
# pbp <- pbp[player==1]

#------------------
# Pull in the Roster Data
#------------------

# select relevant columns from dataset
old_cols <- c('season', 'gsis_id', 'full_name','position',
              'birth_date',  'college', 'height','weight')

new_cols <- c('season', 'player_id', 'player_name', 
              'player_position', 'player_birthdate', 'player_college_name',
              'player_height', 'player_weight')

# select relevant columns from dataset and rename the columns to better names
rosters <- rosters[, ..old_cols]
rosters <- setnames(rosters, old_cols, new_cols)
rosters <- rosters[!is.na(player_id)]

#------------------
# Merge the Play-by-Play and Roster Data
#------------------

merge_stats <- function(player_type){
  
  player_type_cols <- c('season')
  for (i in colnames(rosters)[2:length(new_cols)]){
    i = paste0(player_type, '_',i)
    player_type_cols <- c(player_type_cols, i)
    
  }
  
  rosters <- setnames(rosters, new_cols, player_type_cols)
  
  pbp <- merge.data.table(pbp, rosters, all.x=TRUE, by=c('season', paste0(player_type, '_', 'player_id')))
  rosters <- setnames(rosters, player_type_cols, new_cols)
  
  return(pbp)
}
pbp <- merge_stats('receiver')
pbp <- merge_stats('rusher')
pbp <- merge_stats('passer')

# calculate ages based on season and players birthday
pbp[, receiver_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(receiver_player_birthdate, format='%m/%d/%Y')) / 365)]
pbp[, rusher_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(rusher_player_birthdate, format='%m/%d/%Y')) / 365)]
pbp[, passer_player_age:=as.numeric((as.Date(paste0(season, '-09-01'))-as.Date(passer_player_birthdate, format='%m/%d/%Y')) / 365)]

# filter down the dataset to only rushing and receiving plays
pbp <- pbp[play_type %in% c('run', 'pass', 'kickoff', 'punt', 'field_goal', 'extra_point')]

# save out parquets of the data
arrow::write_parquet(pbp, paste0('/Users/borys/OneDrive/Documents/Github/Daily_Fantasy/Data/OtherData/NFL_FastR/raw_data_', season_pull, '.parquet'))
arrow::write_parquet(rosters, paste0('/Users/borys/OneDrive/Documents/Github/Daily_Fantasy/Data/OtherData/NFL_FastR/rosters_', season_pull, '.parquet'))